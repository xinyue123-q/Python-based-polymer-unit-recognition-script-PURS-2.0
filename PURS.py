#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv, os
from collections import Counter
import pickle as pkl
import os, sys, sparse
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
from sklearn.metrics.pairwise import euclidean_distances
from rdkit.Chem.Draw import IPythonConsole 
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions 
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Draw
import torch
from rdkit import RDLogger
import random
import tqdm
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
from cairosvg import svg2png
import argparse
from IPython.display import display, HTML, SVG, Markdown#可视化模块
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition as rdRGD


# In[37]:


def process_smiles(file_name):
    
    file=open(file_name)
    fileReader=csv.reader(file)
    filedata=list(fileReader)

    smi_list=[]
    name_list0=[]#重名未做处理的名称列表
    name_list=[]#做了重名处理的名称列表
    mol_list = []
    num_list = []
    for i in filedata:
        #Rdkit识别不出顺反，要把'/'与'\'去掉
        if '/'in i[1]:
            i[1]=i[1].replace('/','')
        if '\\'in i[1]:
            i[1]=i[1].replace('\\','')
        mol=Chem.MolFromSmiles(i[1])
        if mol:
            smi=Chem.MolToSmiles(mol)
            num = mol.GetNumAtoms()
            num_list.append(num)
            smi_list.append(smi)
            AllChem.EmbedMolecule(mol)
            mol_list.append(mol)
            adj = Chem.GetAdjacencyMatrix(mol)
        
        #重名的数据会在后面加上重复的个数
            if i[0] in name_list0:
                num = name_list.count(i[0])
                name = i[0]+'-'+str(num)
                name_list.append(name)
            else:
                name_list.append(i[0])
            name_list0.append(i[0])
            
    return smi_list,name_list0,name_list,mol_list,num_list


# In[3]:


def get_bracket_index(s):
##寻找括号及他们的索引
    l_list=[0]#存储“（”的索引
    r_list=[]#存储“）”的索引 
    i_list=[0]#存储所有括号的索引
    i=0
    while i < len(s):
        j=s[i]
        if j =="(":
            i_list.append(i)
            l_list.append(i)
        if j==")":
            i_list.append(i)
            r_list.append(i)
        i=i+1
    l_list=list(reversed(l_list))
    r_list.append(len(s)-1)
    i_list.append(len(s)-1)
    return(l_list,r_list,i_list)


# In[4]:


def smallest(cp_arr,index_arr):
    #寻找最小
    smallest_list=[]
    for i in cp_arr:
        l=i[0]
        r=i[1]
        arr1=index_arr[index_arr>l]
        arr2=arr1[arr1<r]
        if len(arr2)==0:
            smallest_list.append(i)
    return(smallest_list)


# In[5]:


def found_independent_ring_in_same_str2(string_list):
    total_independent_string=[]#真正的独立区域
    for string in string_list:
        independent_string=[]
    #去除字符串里面的斜线
        if "/" in string:
            string=string.replace('/','')
        if "\\" in string:
            string=string.replace('\\','')
    #遍历字符串，找到数字及它的索引
        index_num={}
        num=['0','1','2','3','4','5','6','7','8','9']
        num_list=[]#储存数字的列表
        index_list=[]#储存数字索引的列表
        cp_list=[]#储存具有完整结构的数字对的列表
        #针对存在空num_list的情况
        
        i=0
        #填充index_list
        while i < len(string):    
            j=string[i]
            if j in num:
                index_list.append(i)
                i=i+1
            elif j == "%":
                index_list.append(i)
                i=i+3
            else:
                i=i+1
        
    #找到数字所在的索引，令索引成为字典的键
        if string[1] not in num:
            index_list.append(0)
            index_num[0]=[]
        i=0
        while i < len(string):
            j=string[i]    
            if j in num:
                if j not in num_list:
                    num_list.append(j)
                elif j in num_list:
                    num_list.remove(j)
                index_num[i]=num_list[:]
                cp_like=[]
                for k,v in index_num.items():
                    if v==num_list and k!=i:
                        if k!=index_list[-1]:
                            cp_index=index_list.index(k)+1
                            cp_num=index_list[cp_index]
                            cp_like.append(cp_num)
                if len(cp_like)>0:
                    cp_like.sort()
                    cp=[cp_like[-1],i]
                    cp_list.append(cp)
                i=i+1        
            elif j =="%":
                number=string[i:i]
                if number not in num_list:
                    num_list.append(number)
                elif number in num_list:
                    num_list.remove(number)
                index_num[i]=num_list[:]
                cp_like=[]
                for k,v in index_num.items():
                    if v==num_list and k!=i:
                        if k!=index_list[-1]:
                            cp_index=index_list.index(k)+1
                            cp_num=index_list[cp_index]
                            cp_like.append(cp_num)
                if len(cp_like)>0:
                    cp_like.sort()
                    cp=[cp_like[-1],i]
                    cp_list.append(cp)
                i=i+3        
            else:
                i=i+1  
               
        #消除内部括号
        new_cp_list=[]
        cp_arr=np.array(cp_list)
        for i in cp_arr:
            cp_arr2=cp_arr[cp_arr[:,0]<i[0]]
            cp_arr3=cp_arr2[cp_arr2[:,1]>i[1]]
            if len(cp_arr3)==0:
                i=i.tolist()
                new_cp_list.append(i)
        cp_list=new_cp_list
    
        #找到准独立区域
        be_string={}#用于储存准独立区域的列表
        for i in cp_list:   
            s=""
            name=str(i[0])+"-"+str(i[1])
            if i==cp_list[0] and index_num[i[1]]==[]:
                if i[1]<len(string)-2:
                    if string[i[1]+1]=="=" and string[i[1]+2]=="O":
                        s=s+string[i[0]-1:i[1]+3]
                        
                else:
                    s=s+string[i[0]-1:i[1]+1]
                be_string[name]=[s,[0,i[1]]]
            else:
                if string[i[1]]!="%":
                    if i[1]<len(string)-2:
                        if string[i[1]+1]=="=" and string[i[1]+2]=="O":
                            s=s+string[i[0]-1:i[1]+3]
                    else:
                        s=s+string[i[0]-1:i[1]+1]
                be_string[name]=[s,i]
                if string[i[1]]=="%":
                    s=s+string[i[0]-1:i[1]+3]
                    be_string[name]=[s,i]
        
        #给划分出来的独立结构加括号
        for k,v in be_string.items():
            v_list=[v[0]]
            v[0]=add_bracket(v_list)[0]
            v[0]="C"+v[0]
    
        #确认是否是真正的独立区域
        real_string={}
        for k,v in be_string.items():
            mol = Chem.MolFromSmiles(v[0])
            if mol:
                real_string[k]=v
    
        #剩余的外一层区域也是独立区域
        first=0
        last=len(string)
        outside_num=[]
        for k,v in real_string.items():
            outside_num.append(v[1][0])
            outside_num.append(v[1][1])

        outside_num.sort()
        if len(outside_num)>0:
            if outside_num[0]!=0:
                outside_str=string[0:outside_num[0]]+"(C)"
                i=1
                while i<len(outside_num)-1:
                    j=outside_num[i]
                    k=outside_num[i+1]
                    if string[j]!="%":
                        outside_str=outside_str+"(C)"+string[j:k+1]
                    elif string[j]=="%":
                        outside_str=outside_str+"(C)"+string[j+3:k+1]
                    i=i+2

            if outside_num[0]==0:
                outside_str=""
                i=1
                while i<len(outside_num)-1:
                    j=outside_num[i]
                    k=outside_num[i+1]
                    if string[j]!="%":
                        outside_str=outside_str+"(C)"+string[j:k+1]
                    elif string[j]=="%":
                        outside_str=outside_str+"(C)"+string[j+3:k+1]
                    i=i+2
        
            if outside_num[-1]!=last:
                n=outside_num[-1]
                m=string[n]
                if m!="%":
                    outside_str=outside_str+"(C)"+string[n:last]
                if m=="%":
                    outside_str=outside_str+"(C)"+string[n+3:last]
            for k,v in real_string.items():
                independent_string.append(v[0])
    #判断是否具有独立结构被分出，没有就记录原来的结构
        if len(independent_string)==0:
            total_independent_string.append(string)
        else:
            mol=Chem.MolFromSmiles(outside_str)
            if mol:
                for i in independent_string:
                    total_independent_string.append(outside_str)
            for i in independent_string:
                        total_independent_string.append(i)
    
    return(total_independent_string)





def make_smi(smi):
    #输入是字符串，给字符串去斜线，去括号，使字符串能被RDKIT识别
    if len(smi)>1:
        if smi[0]=="(" and smi[-1]==")":
            smi=smi[1:-1]
    if "/" in smi:
        smi=smi.replace('/','')
    if "\\" in smi:
        smi=smi.replace('\\','')
    if "-" in smi:
        smi=smi.replace('-','')
    if smi[0]=="=":
        smi=smi[1:]
    return(smi)





def add_bracket(string):
    right_bracket=0
    left_bracket=0
    if string[0]==")":
        string=string[1:]
    if string[0]=="=":
        string="C"+string
    for j in string:
        if j =="(":
            right_bracket=right_bracket+1
        if j==")":
            left_bracket=left_bracket+1
    if right_bracket>left_bracket:
        n=right_bracket-left_bracket
        while n>0:
            string=string+")"
            n=n-1
    if right_bracket<left_bracket:
        m=left_bracket-right_bracket
        while m>0:
            string="("+string
            m=m-1
    if len(string)>2:
        if string[0]=="(" and string[1]==")":
            string=string[2:]
        if string[0]==")":
            string=string[1:]
    
    return(string)
        


# In[8]:


def if_mol(smi_list):
    #判断字符串是否是正确的SMILES码
    smi_list=[]
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            wrong_list.append(smi)
    return(smi_list)
        


# In[9]:


def found_independent_ring_in_same_str1(independent_ring_string0):
    real_independent_ring=[]#独立区域里面可能不止一个独立环，找出每个这样的环
    
    num=['1','2','3','4','5','6','7','8','9']
    for i in independent_ring_string0:
        #print("上层结构：")
        #print(i)
        j=0
        num_list=[]
        ring_list=[]
        cp_list=[]
        index1=0
        while j<len(i):
            k=i[j]
            if k in num:
                num_list.append(k)
                j=j+1
                if len(num_list)==2*len(set(num_list)):
                    num_list=[]
                    #防止碳氧双键被剪掉
                    if j <= (len(i)-2):
                        if i[j+1]=="=" and i[j+2]=="O":
                            cp_list.append([index1,j+3])
                        else:
                            cp_list.append([index1,j])
                    else:
                        cp_list.append([index1,j])
                    index1=j
            elif k == "%":
                number=str(i[j+1])+str(i[j+2])
                num_list.append(number)
                j=j+3
                if len(num_list)==2*len(set(num_list)):
                    num_list=[]
                    if j <= (len(i)-2):
                        if i[j+1]=="=" and i[j+2]=="O":
                            cp_list.append([index1,j+3])
                        else:
                            cp_list.append([index1,j])
                    else:
                        cp_list.append([index1,j])
                    index1=j
            else:
                j=j+1
        if cp_list !=[]:
            cp_list[-1][1]=len(i)
            for k in cp_list:
                f=k[0]
                l=k[1]
                s=i[f:l]
                ring_list.append(s)
        if len(ring_list)==0:
            real_independent_ring.append(i)
        if len(ring_list)!=0:
            #print("下层结构：")
            for g in ring_list:
                #print(g)
                real_independent_ring.append(g)
    return(real_independent_ring)


# In[10]:


def link_c(string):
    #对切出来的片段加连接键
    num_list=['1','2','3','4','5','6','7','8','9','%']
    #给环补上连接碳
    if len(string)>1:
        #这里人为筛选掉了太小的片段
        if string[0]=="C" and string[1] in num_list:
            string="C"+string
    return(string)


# In[11]:


def found_location_in_DataFrame_double(DataFrame,key): 
    #找到一个值在Data_Frame里面的位置，并且返回行数和列数
    #当值是一个列表时
    hang=DataFrame.shape[0]-1#DataFrame的行数
    location=[]
    while hang >=0:
        lie=DataFrame.shape[1]-1#DataFrame的列数
        while lie >=0:
            cp=DataFrame[lie][hang]
            if len(cp)>1:#判断元素是否为列表
                if cp==key:
                    location.append([hang,lie])#[行，列]
            lie=lie-1
        hang=hang-1
    return(location)



def structure_DataFrame(c_list,smallest_list,r_list,l_list):
    str_DataFrame=pd.DataFrame(index=np.arange(len(smallest_list)),columns=np.arange(len(c_list)))
    str_DataFrame[:]='n'
    #将最小独立区域输入至DataFrame的第一列
    i=0 
    while i < len(smallest_list):
        j=smallest_list[i]
        str_DataFrame[0][i]=j
        i=i+1
    
    #依次寻找独立区域
    list_2=[]
    list_1=smallest_list
    k=0
    
    while k<(len(c_list)-1):
        for u in list_1:
            if len(u)== 2:
                if u[1] !=c_list[-1][1]:
                    plus_u=int(c_list.index(u))+1#前一个左括号
                    u_plus=c_list[plus_u]
                    while (u_plus[1]<u[1])and(u_plus[1]!=c_list[-1][1]):
                        plus_plus_u=c_list.index(u_plus)+1
                        u_plus=c_list[plus_plus_u]
                    if u_plus[1]<u[1]:
                        list_2.append("n")
                    if u_plus[1]>u[1]:
                        list_2.append(u_plus)
                if u[1] ==c_list[-1][1]:
                    list_2.append("n")
            elif len(u)==1:
                list_2.append("n")
        j=0
        for n in list_2:
            str_DataFrame[k+1][j]=n
            j=j+1
        k=k+1#写入列的指数加1
        list_1=list_2#更新列表
        list_2=[]
    return(str_DataFrame)



def pairing(smiles,index_list,l_list,r_list):
    ##对括号进行匹配
    i=0
    cp_ed=[]#存储已配对右括号
    c_list=[]#存储所有已配对括号
    while i < len(l_list):
        l=l_list[i]
        j=0
        try:
            while (r_list[j] < l) or (r_list[j] in cp_ed):
                j=j+1
        except IndexError:
            j=j-1
        r=r_list[j]
        cp_ed.append(r)
        cp=[l,r]
        c_list.append(cp)
        i=i+1
    #将字符串第一个数字索引和最后一个数字索引存入cp_list
    first=0
    last=len(smiles)
    if first not in index_list:
        if last not in index_list:
            c_list.append([first,last])
    return(c_list)




def found_location_in_DataFrame_single(DataFrame,key): 
    #找到一个值在Data_Frame里面的位置，并且返回行数和列数
    #当值是一个数时
    hang=DataFrame.shape[0]-1#DataFrame的行数
    location=[]
    while hang >=0:
        lie=DataFrame.shape[1]-1#DataFrame的列数
        while lie >0:
            cp=DataFrame[lie][hang]
            if cp==key:
                location.append([hang,lie])#[行，列]
            lie=lie-1
        hang=hang-1
    return(location)




def delete_free_radical_in_index_data(index_data):
    index_data2={}
    for k,v in index_data.items():
        if '[C]'in v[1]:
            v2=v[1].replace('[C]','C')
            index_data2[k]=[v[0],v2]
        else:
            index_data2[k]=v
    return(index_data2)



def bratch_in_string(s):
    num_list=['0','1','2','3','4','5','6','7','8','9','%']
    br_list=[]
    #对于在字符串前面的支链
    index1=0
    bratch=""
    i=0
    while i < len(s):
        n=s[i]
        if n in num_list:
            bratch=s[0:i]
            break
        i=i+1
    
    #在前面的支链
    if len(bratch)>=6:
        if "[C]" not in bratch:
            s1=s[i-1:]
            br_list.append(bratch)
        else:
            s1=s
    else:
        s1=s
    #对于在字符串后面的支链
    j=len(s1)-1
    bratch2=''
    while j>0:
        k=s1[j]
        if k in num_list:
            bratch2=s1[j+1:]
            break
        j=j-1
    if len(bratch2)>=6:
        if "[C]" not in bratch2:
            s2=s1[:j+1]
            br_list.append(bratch2)
        else:
            s2=s1
    else:
        s2=s1
    return(s2,br_list)
    



def bratch_select(index_data0,index_data,str_df,index_cp,bratch_cp,smiles):
    cp_to_delete=[]
    cp_to_repair={}
    for i in index_cp:
        if i in bratch_cp:
            name=str(i[0])+"-"+str(i[1])
            location=found_location_in_DataFrame_double(str_df,i)
            string=index_data0[name][1]
            for j in location:
                plus=str_df[j[1]+1][j[0]]
                if plus in index_cp:
                    plus_name=str(plus[0])+"-"+str(plus[1])
                    plus_string=index_data0[plus_name][1]
                    inner_num=index_data0[plus_name][0]
                    if "CCCC" in plus_string:
                        inner_num.remove(i[0])
                        inner_num.remove(i[1])
                        cp_to_repair[plus_name]=inner_num
                        cp_to_delete.append(name)                        
    for key,v in cp_to_repair.items():
        v=inner_num
        if len(inner_num)>=1:
            inner_num.sort()
            slice_s=""
            i=0
            while i < (len(inner_num)-1):
                j=inner_num[i]
                k=inner_num[i+1]
                s1=smiles[j:k+1]
                slice_s=slice_s+s1
                if i <(len(inner_num)-2):
                    l=inner_num[i+2]
                    s2='C'
                    slice_s=slice_s+s2
                i=i+2
            v2=[int(key[0]),int(key[2])]
            index_data[key]=[v2,slice_s]
    #for i in cp_to_delete:
        #print(i)
        #print(index_data[i])
        
    return(index_data)
    
 





def make_con(index_data,index_cp,br):
    index_data2={}
    error=[]
    error_k=[]
    for k,v in index_data.items():
        mol=Chem.MolFromSmiles(v[1])
        if mol:
            smi = Chem.MolToSmiles(mol)
            index_data2[k]=[v[0],smi]
        if not mol:
            error.append(v[0])
            error_k.append(k)
    for i in error:
        if i in index_cp:
            index_cp.remove(i)
    for i in error_k:
        if i in br:
            del br[i]
            
    return(index_data2,index_cp,br)
   


# In[21]:


def bratch_amend(br):
    br2={}
    for k,v in br.items():
        v2=[]
        for i in v:
            if '(C)'in i:
                i=i.replace('(C)','')
            if 'c'in i:
                i=i.replace('c','C')
            v2.append(i)
        br2[k]=v2
    return(br2)




def found_end_point_neighbour(smiles,neighbor_data,index_data):
    #给单体两端的基元匹配最近邻

    #1.寻找两个端点
    left_end=[]
    right_end=[]
    l_or_r=[]
    for k,v in neighbor_data.items():
        if "br" not in k:
            m=index_data[k]
            if '[C]'in m[1]:
                if v['right_neighbor']=={}:
                    left_end.append(m)
                
                if v['left_neighbor']=={}:
                    right_end.append(m)
                
                else:
                    l_or_r.append(m)
                
    left_name="left_name"
    left_smiles="left_smiles"
    right_name="right_name"
    right_smiles="right_smiles"
    #两个端点都找到
    if len(left_end)>=1 and len(right_end)>=1:
        left=left_end[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=left_end[0][1]
        right=right_end[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=right_end[0][1]
        #2.左端点的右近邻是右端点
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data

    #只找到左端点    
    elif len(left_end)==1 and len(l_or_r)==1:
        left=left_end[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=left_end[0][1]
        right=l_or_r[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=l_or_r[0][1]
        #2.左端点的右近邻是右端点
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
        
    #只找到右端点    
    elif len(right_end)==1 and len(l_or_r)==1:
        right=right_end[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=right_end[0][1]
        left=l_or_r[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=l_or_r[0][1]
        #2.左端点的右近邻是右端点
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
       
    #两个端点都不确定
    elif len(l_or_r)==2:
        right=l_or_r[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=l_or_r[0][1]
        left=l_or_r[1][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=l_or_r[1][1]
        #2.左端点的右近邻是右端点
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
    
    #一个端点都没有找到
    elif len(l_or_r)==1:
        right=l_or_r[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=l_or_r[0][1]
        left=l_or_r[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=l_or_r[0][1]
        #2.左端点的右近邻是右端点
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
    #两个都是左端
    elif len(right_end)==2:
        num1=right_end[0][0][1]
        num2=right_end[1][0][1]
        if num1>num2:
            right=right_end[0][0]
            right_name=str(right[0])+"-"+str(right[1])
            right_smiles=right_end[0][1]
            left=right_end[1][0]
            left_name=str(left[0])+"-"+str(left[1])
            left_smiles=right_end[1][1]
        if num1<num2:
            right=right_end[1][0]
            right_name=str(right[0])+"-"+str(right[1])
            right_smiles=right_end[1][1]
            left=right_end[0][0]
            left_name=str(left[0])+"-"+str(left[1])
            left_smiles=right_end[0][1]
        #注意两个
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["left_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data

    
    return(neighbor_data)



def found_neighbor(str_df,index_data,index_cp):
    neighbor_data={}
    for i in index_cp:
        location=found_location_in_DataFrame_double(str_df,i)#找到元素在str_df中的位置
        right_neighbour=[]
        left_neighbour=[]
        for j in location:            
            hang=j[0]
            if j[1]<(str_df.shape[1]-1):
                right_lie=j[1]+1
                right=str_df[right_lie][hang]
                m=str_df.shape[1]-1
                while (right not in index_cp) and (right_lie!=m):
                    right_lie=right_lie+1
                    right=str_df[right_lie][hang]
                if right in index_cp:
                    right_name=str(right[0])+"-"+str(right[1])
                    right_neighbour.append(right_name)
            if j[1]>0:                                   
                left_lie=j[1]-1
                left=str_df[left_lie][hang]
                while (left not in index_cp) and (left_lie>0):
                    left_lie=left_lie-1
                    left=str_df[left_lie][hang]
                if left in index_cp:
                    left_name=str(left[0])+"-"+str(left[1])
                    left_neighbour.append(left_name)
        
        right_list=[]
        for r in right_neighbour:
            if r not in right_list:
                right_list.append(r)
      
        left_list=[]
        for l in left_neighbour:
            if l not in left_list:
                left_list.append(l)
        
        name=str(i[0])+"-"+str(i[1])
        neighbor={}
        right_neighbor={}
        left_neighbor={}
        self=index_data[name][1]

        for item in right_list:
            right_neighbor[item]=index_data[item][1]
        for item in left_list:
            left_neighbor[item]=index_data[item][1]
        if right_list==[] and left_list==[]:
            right_neighbor[name]=index_data[name][1]
            left_neighbor[name]=index_data[name][1]
        
        #匹配支链
        #if len(br[name])!=0:
            #count=len(br[name])
            #i=0
            #while i<count:
                #br_name=name+"-br-"+str(i)
                #left_neighbor[br_name]=br[name][i]
                #i=i+1
        
        neighbor["right_neighbor"]=right_neighbor
        neighbor["left_neighbor"]=left_neighbor
        neighbor["self"]=self
        neighbor_data[name]=neighbor
        
        neighbor_data2=neighbor_data.copy()
    #将支链写入主结构
    #for k,v in br.items():
        #if len(v)!=0:
            #count=len(v)
            #nei={}
            #nei[k]=index_data[k][1]
            #i=0
            #while i<count:
                #br_name=k+"-br-"+str(i)
                #data={}
                #data['self']=v[i]
                #data['right_neighbor']=nei
                #data['left_neighbor']={}
                #neighbor_data[br_name]=data
                #i=i+1
    return(neighbor_data)            


def add_bratch(index_data,br):
    for k,v in br.items():
        if len(v)>0:
            for idx,i in enumerate(br):
                name = k+'-br-'+str(idx)
                index_data[name]=i
    return index_data


def get_atom_num(smiles,inner_dist):
    #返回SMILES码中每个字符所代表的原子（如果他们确实代表了某个原子的话）对应的原子序数
    count = 0
    num_list = []
    problem_atom = ['e','i','a','g','l','r']
    atom_index = {
        'e':['S','G'],
        'i':['L','S','B'],
        'a':['N','C'],
        'g':['M'],
        'l':['C'],
        'r':['B']
                 }
    end_atom = []
    end_index = []
    #生成原子序数列表
    for idx,i in enumerate(smiles):
        if i.isalpha():
            if i in problem_atom:
                if idx>0:
                    j = smiles[idx-1].upper()
                    k = atom_index[i]
                    if j in k:
                        num_list.append(count-1)
                    else:
                        num_list.append(count)
                        count = count+1
                else:
                    num_list.append(count)
            elif i == 'H':
                num_list.append(-1)
            else:
                num_list.append(count)
                count = count+1
        else:
            num_list.append(-1)
        if i == '[' :
            if smiles[idx+1] == 'C':
                if smiles[idx+2] == ']':
                    end_atom.append(idx+1)
    
    
    end_atom_num = []
    for i in end_atom:
        index = num_list[i]
        end_atom_num.append(index)
        
    #给每一个基元一个原子序数列表
    for k,v in inner_dist.items():
        #首先生成字符串索引列表
        i = 0
        n = []
        while i < len(v):
            j = list(range(v[i],v[i+1]+1))
            n = n+j
            i = i+2
            
        #然后根据索引找到每个Polymer-unit所包含的原子序号
        atom_list = []
        for index in n:
            num = num_list[index]
            if num >= 0:
                atom_list.append(num)
            if num in end_atom_num:
                end_index.append(k)
        atom_set = list(set(atom_list))
        
        #最后将inner_dist的值替换为所包含的原子的序数
        inner_dist[k]=atom_set
    
    return inner_dist,end_atom_num,end_index



def get_pair_atom(total_neighbor_data,total_inner_dist):
    pair_atom_dist = {}
    count = 0 
    for key,value in total_neighbor_data.items():
    
        count = count + 1
        polymer = total_neighbor_data[key]
        inner_dist = total_inner_dist[key]
        atom_dist = {'pair_atom':[],'pair_index':[]}
        pair_atom_dist[key] = atom_dist
    
        for k,v in polymer.items():
                     
            polymer_unit = polymer[k]
            inner_num = inner_dist[k]
            inner_num.sort()        
            link_atom = inner_num[0]-1
            r_nei = polymer_unit['right_neighbor']
            l_nei = polymer_unit['left_neighbor']
            if r_nei == 0:
                if l_nei == 0:
                    print('There is connected problem in {} polymer at {} unit'.format(key,k))
            if r_nei != 0:
                for i in r_nei.keys():
                    inner_num_r = inner_dist[i]
                    if link_atom in inner_num_r:
                        pair_atom = [inner_num[0],link_atom]
                        pair_index = [k,i]
                        atom_dist['pair_atom'].append(pair_atom)
                        atom_dist['pair_index'].append(pair_index)
                    
        #num_pair = len(atom_dist['pair_atom'])
        #print('{}单体,一共{}个基元,找到{}个原子对'.format(key,len(polymer),num_pair))
    return pair_atom_dist



def get_adj(ring_total_list,total_neighbor_data,name_list):
    ###生成连接矩阵(矩阵维度根据原子数)
    ring_series=pd.Series(ring_total_list)

    matrix_list=[]
    for idx,i in enumerate(name_list):
        name = name_list[idx]
        n_max = len(total_neighbor_data[name])
        data = total_neighbor_data[i]
        #生成连接矩阵
        matrix = np.zeros((n_max,n_max),dtype = np.int8)
        self_list = []
        for k,v in data.items():
            self_list.append(k)
        j = 0
        while j < len(self_list):
            k = self_list[j]
            data2 = data[k]
            for k,v in data2['right_neighbor'].items():
                index = self_list.index(k)
                matrix[j,index] = 1
                matrix[index,j] = 1
            for k,v in data2['left_neighbor'].items():
                index = self_list.index(k)
                matrix[j,index] = 1
                matrix[index,j] = 1
            j=j+1

        matrix_list.append(matrix)
    return matrix_list



def get_pu_index(total_neighbor_data,ring_total_list,name_list):
    #每个数据的每个基元在基元列表中的索引编号
    pu_index={}
    for key,value in total_neighbor_data.items():
        index={}
        pu_index[key] = index 
        for k,v in value.items():
            locat = ring_total_list.index(v['self'])
            index[k]=locat
            
    pu_index2 = {}
    for key,values in pu_index.items():
        if key in name_list2:
            pu_index2[key]=values
            
    return pu_index2


def get_node_index(total_neighbor_data,ring_total_list):
    node_index=[]
    node_num_list=[]
    for key,value in total_neighbor_data.items():
        index=[] 
        for k,v in value.items():
            locat = ring_total_list.index(v['self'])
            index.append(locat)
        node_index.append(index)
        node_num_list.append(len(index))
    return node_index, node_num_list



def get_pu_dict(total_neighbor_data,ring_total_list):
    pu_index={}
    for key,value in total_neighbor_data.items():
        index={}
        pu_index[key] = index 
        for k,v in value.items():
            locat = ring_total_list.index(v['self'])
            index[k]=locat
    return pu_index



def bondFeatures2(pos, bid1, bid2, mol, rings):
    
    bondpath = Chem.GetShortestPath(mol, bid1, bid2)
    bonds = [mol.GetBondBetweenAtoms(bondpath[t], bondpath[t + 1]) for t in range(len(bondpath) - 1)]
    pos1 = pos[bid1]
    pos2 = pos[bid2]
    dis = np.linalg.norm(pos1-pos2)
    a1 = mol.GetAtomWithIdx(bid1)
    a2 = mol.GetAtomWithIdx(bid2)
    type1 = a1.GetAtomicNum()
    type2 = a2.GetAtomicNum()
    samering = 0
    for ring in rings:
        if bid1 in ring and bid2 in ring:
            samering = 1

    if len(bonds)==1:
        v1 = [dis,type1,type2]
        v2 = to_onehot(str(bonds[0].GetStereo()), ['STEREOZ', 'STEREOE','STEREOANY','STEREONONE'])[:3]
    else:
        v1 = np.zeros(3)
        v2 = np.zeros(3)
        
    return np.concatenate([v1, v2], axis=0)



#对浮点数四舍五入
def get_int(num):
    num1, num2 = str(num).split('.')
    if float(str(0) + '.' + num2) >= 0.5:
        return(int(num1) + 1)
    else:
        return(int(num1))



def to_onehot(val, cat):

    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec

def rigin_type_classify(cp_list,smiles,smallest_r,str_DataFrame):
    independent_cp0=[]
    dependent_cp0=[]
    bratch0={}
    bratch_cp0=[]
    i=0#列指标的循环迭代数
    num=['1','2','3','4','5','6','7','8','9','10']
    while i<(len(cp_list)):
        j=0
        while j<len(smallest_r):
            h=str_DataFrame[i][j]#i列j行df的元素--某个括号对
            out_judge=True
            if i ==len(cp_list)-1:
                out_judge = False
            else:
                h_out=str_DataFrame[i+1][j]#这个括号对外一行括号对
                if len(h_out)==2:#说明df的外面确实存在括号对
                    string_out=smiles[h_out[0]:(h_out[1]+1)]#这个括号外面的SMILES码
            j=j+1
            if len(h)==2:#说明df的这个位置存在括号对
                num_list=[]#存放SMILES码的切片中包含的数字
                string=smiles[h[0]:(h[1]+1)]#这个括号对里面的SMILES码
                k=0
                #遍历括号对包含的SMILES码，找到其中包含的所有数字
                while k < len(string):
                    e=string[k]
                    if e=="%":
                        u=str(string[k+1])+str(string[k+2])
                        num_list.append(u)
                        k=k+3
                    if e in num:
                        num_list.append(e)
                        k=k+1
                    else:
                        k=k+1
                diff=len(num_list)-len(set(num_list))
                if len(num_list)==0:#这个括号对里面没有环结构
                    dependent_cp0.append(h)
                elif len(num_list)==1:
                    dependent_cp0.append(h)#括号对里面只有一个数字，肯定不是独立的
                elif len(num_list)>1:
                    judge=False
                    for n in num_list:
                        r=num_list.count(n)%2
                        if r==1:
                            judge=True
                    if judge==False:
                        independent_cp0.append(h)
                    if judge==True:
                        dependent_cp0.append(h)
                
        i=i+1#开始处理下一列
    
    return(independent_cp0,dependent_cp0)

def make_con(index_data,index_cp):
    index_data2={}
    error=[]
    error_k=[]
    for k,v in index_data.items():
        mol=Chem.MolFromSmiles(v[1])
        if mol:
            smi = Chem.MolToSmiles(mol)
            index_data2[k]=[v[0],smi]
        if not mol:
            error.append(v[0])
            error_k.append(k)
    for i in error:
        if i in index_cp:
            index_cp.remove(i)
    #for i in error_k:
    #    if i in br:
    #        del br[i]
            
    return(index_data2,index_cp)

#生成polymer-unit的MACCS
def get_MACCS(ring_total_list,N):
    MACCS_dict={}
    index=torch.LongTensor(random.sample(range(167), N))
    for idx,i in enumerate(ring_total_list):
        molecule = Chem.MolFromSmiles(i)
        if not molecule:
            j = i.upper()
            molecule = Chem.MolFromSmiles(j)
        MACCS_fp = MACCSkeys.GenMACCSKeys(molecule)
        MACCS = MACCS_fp.ToBitString()
        MACCS = np.array(list(MACCS),dtype=int)
        MACCS = torch.tensor(MACCS,dtype=torch.float)
        if N != False:
            MACCS_select = torch.index_select(MACCS, 0, index)
            MACCS_select = MACCS_select.numpy().tolist()
            MACCS_dict[idx]=MACCS_select
        if N == False:
            MACCS_dict[idx]=MACCS.numpy().tolist()
    
        #print(MACCS_select)
    return MACCS_dict


def get_node_feature(name_list,MACCS_dict,pu_index,node_num_list):
    node_feature_dist={}
    for idx, i in enumerate(name_list):
        node_num = node_num_list[idx]
        index = []
        for k,v in pu_index[i].items():
            index.append(int(v))
        feature_list = []
        for j in index:
            f = MACCS_dict[j]
            feature_list.append(f)
            
        feature = torch.tensor(feature_list)
        node_feature_dist[i] = feature
    return node_feature_dist

#生成每个单体的边特征值
def get_edge_feature(smi_list,name_list,matrix_list,pu_index,pair_atom_dist,total_inner_dist):
    dim_edge = 8
    writer = Chem.SDWriter('polymer.sdf')

    for i, smi in enumerate(smi_list):
       mol = Chem.MolFromSmiles(smi)
       mol.SetProp('_Name',name_list[i])
       writer.write(mol)
    writer.close()

    structure = Chem.SDMolSupplier('polymer.sdf',removeHs=False)

    total_edge_list=[]
    edge_num_list = []
    for i, mol in enumerate(structure):
        if not mol:
            print(name_list[i])
         
        pos = mol.GetConformer().GetPositions()
        name = name_list[i]
        num_index = list(pu_index[name].values())#节点的种类
        num_node = len(num_index)#每个单体中节点的个数
        node_name = list(pu_index[name].keys())#节点的编号
        pair_atom = pair_atom_dist[name]
        smi = Chem.MolToSmiles(mol)
        if '.' in Chem.MolToSmiles(mol): continue
        n_atom = mol.GetNumAtoms()
        rings = mol.GetRingInfo().AtomRings()
        ##edge DE
        adj = matrix_list[i]
        node_bond = get_int(sum(sum(adj))/2)
        pair_index = pair_atom['pair_index']
        atom_pair = pair_atom['pair_atom']
        num_pair = len(atom_pair)
    
    
        edge_list = []
        for j in range(num_node):
            for k in range(num_node):
                if adj[j,k]==1:
                    j_name=node_name[j]
                    k_name=node_name[k]
                    j_index=num_index[j]
                    k_index=num_index[k]
                    pair1=[j_name,k_name]
                    pair2=[k_name,j_name]
                    if pair1 in pair_index:   
                        locat = pair_index.index(pair1)
                        edge = np.zeros([8])
                        pair = atom_pair[locat]
                        edge[:6] = bondFeatures2(pos,pair[0],pair[1], mol, rings)
                        edge[6] = j_index*0.01
                        edge[7] = k_index*0.01
                        edge_list.append(edge.tolist())
                        edge_list.append(edge.tolist())
                    elif pair2 in pair_index:
                        locat = pair_index.index(pair2)
                        edge = np.zeros([8])
                        pair = atom_pair[locat]
                        edge[:6] = bondFeatures2(pos,pair[0],pair[1], mol, rings)
                        edge[6] = j_index*0.01
                        edge[7] = k_index*0.01
                        edge_list.append(edge.tolist())
                        edge_list.append(edge.tolist())
                    else:
                        #print("Index_error: {} no found".format(pair1))
                        edge = np.zeros([8])
                        edge_list.append(edge.tolist())
                        edge_list.append(edge.tolist())
        edge = torch.tensor(edge_list,dtype=torch.float)
        total_edge_list.append(edge)
        edge_num_list.append(len(edge_list))
    return total_edge_list, edge_num_list

def edge_index(matrix_list):
    
    senders_list=[]
    receivers_list=[]
    edge_num_list=[]
    for i in matrix_list: 
        i=torch.tensor(i,dtype=torch.long)
        nozero=torch.nonzero(i)
        edge_num_list.append(len(nozero))
        nozero2=nozero[:,[1,0]]
        senders=torch.flatten(nozero)
        receivers=torch.flatten(nozero2)
        senders_list.append(senders)
        receivers_list.append(receivers)
        
    return senders_list,receivers_list,edge_num_list

def add_bratch_to_list(v,bratch_list):
    br_m = Chem.MolFromSmiles(v)
    if br_m:
        num = br_m.GetNumAtoms()
        if num > 3:
            bratch_list.append(v)                    
    else:
        print("not mol")
        print(v)
        
def hight_atom(mol,highlight_atoms):
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel',str(atom.GetIdx()))
    image_data = BytesIO()
    view = rdMolDraw2D.MolDraw2DSVG(500, 500)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    option = view.drawOptions()
    for atom in mol.GetAtoms():
        option.atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx() + 1)
    view.DrawMolecule(tm, highlightAtoms=highlight_atoms)
    view.FinishDrawing()
    svg = view.GetDrawingText()
    SVG(svg.replace('svg:', ''))
    svg2png(bytestring=svg, write_to='test.png')
    img = Image.open('test.png')
    #img.save(image_data, format='PNG')
    display(img)
    
def get_bratch_dist(smi_list,name_list):
    total_bratch_dist = {}
    link_symbol_to_delete = [f"[*:{i}]" for i in range(20)]
    for idx,smi in enumerate(smi_list):
        name = name_list[idx]
        bratch_list = []
        inner_list = []
        bratch_dist = {}
        core_dist ={}
        m = Chem.MolFromSmiles(smi)
        for atom in m.GetAtoms():
            atom.SetProp('atomLabel',str(atom.GetIdx()))
        fig = Draw.MolToImage(m, size=(800,800), kekulize=True)
        #显示分子结构
        #display(fig)
        core = MurckoScaffold.GetScaffoldForMol(m)
        core_smi = Chem.MolToSmiles(core)
        flag =m.HasSubstructMatch(core)
        if flag:
            C_atomids = m.GetSubstructMatches(core)
            C_atomids = list(C_atomids[0])
        core_dist['smiles'] = core_smi
        core_dist['inner'] = C_atomids
        bratch_dist['core'] = core_dist
        m_core = [m, core]
        res, unmatched = rdRGD.RGroupDecompose([core], [m], asSmiles=True)
        v_list = []
        for k,v in res[0].items():
            if k != 'Core':
                for i in link_symbol_to_delete:
                    if i in v:
                        v=v.replace(i,'')
                if '.' in v:
                    cut_index = v.index('.')
                    v1 = v[cut_index+1:]
                    v2 = v[:cut_index]
                    v_list.append(v1)
                    v_list.append(v2)
                else:
                    v_list.append(v)
        for v in v_list:
            #add_bratch_to_list(v,bratch_list)
            br_m = Chem.MolFromSmiles(v)
            if br_m:
                num = br_m.GetNumAtoms()
                if num > 1:
                    bratch_list.append(v)
            else:
                print("mol error")
                
        bratch_list = list(set(bratch_list)) 
        atomids_set = []

        for idx,i in enumerate(bratch_list):
            br_dist ={'smiles':i,'inner':[]}
            br_m = Chem.MolFromSmiles(i)
            flag =m.HasSubstructMatch(br_m)
            if flag:
                atomids_list = []
                atomids = m.GetSubstructMatches(br_m)
                for n in atomids:
                    not_bratch = list(set(n) & set(C_atomids))
                    not_br_num = len(not_bratch)
                    if not_br_num == 0:
                        atom_num = len(n)
                        atomids_list.append(n)

                for j in atomids_list:
                    same = 0
                    for k in atomids_list:
                        if len(list(set(k) & set(j))) == 0:
                            same = same + 1
                    if same == (len(atomids_list)-1):
                        atomids_set.append(j)
                        br_dist['inner'].append(list(j))
            else:
                print("No branch chain matches")
                print(i)
            bratch_dist[idx]=br_dist
        delete_inner_list = []
        for i in atomids_set:
            same = 0
            for j in atomids_set:
                if len(list(set(i) & set(j))) != 0:
                    if len(i)>len(j):
                        delete_inner_list.append(list(j))
                    if len(i)<len(j):
                        delete_inner_list.append(list(i))

        for k,v in bratch_dist.items():
            if k != 'core':
                inner_list = v['inner']
                for i in inner_list:
                    if i in delete_inner_list:
                        inner_list.remove(i)
                v['inner']  = inner_list
                
        total_bratch_dist[name]=bratch_dist    
    return total_bratch_dist

def get_pu(smi_list,name_list):
    ###主程序1--带有邻接特征
    ring_total_list=[]
    error_find_independent_str1=[]
    total_neighbor_data={}
    total_inner_dist={}
    total_end_atom_pair = {}
    n=0#总循环数
    while n < len(smi_list):   
    #将SMILES拆分为单个元素
        smiles=smi_list[n]
        name=name_list[n] 
    #找到所有括号的索引
        from_get_bracket_index=get_bracket_index(smiles)
        left_index_list=from_get_bracket_index[0]#存储“（”的索引
        right_index_list=from_get_bracket_index[1]#存储“)”的索引
        index_list=from_get_bracket_index[2]#存储所有括号的索引
        ##对括号进行匹配
        cp_list=pairing(smiles,index_list,left_index_list,right_index_list)
        cp_arr=np.array(cp_list)#将储存括号对的列表转化为np.array
        index_arr=np.array(index_list)#将存储所有括号索引的列表转化为np.array
    
        #识别最小括号
        smallest_r=smallest(cp_list,index_arr)
        
        #以最小独立区域为起点，DataFrame。第一列是最小独立区域，第二列是外一层，第三列是外二层
        str_df=structure_DataFrame(cp_list,smallest_r,right_index_list,left_index_list)
    
        #创建三个列表，将独立括号区域进行分类，分成独立环，支链，非独立环三种
        independent_cp_and_dependent_cp=rigin_type_classify(cp_list,smiles,smallest_r,str_df)
        independent_cp=independent_cp_and_dependent_cp[0]
        dependent_cp=independent_cp_and_dependent_cp[1]
       #创建一个所有非最小括号的括号字典
        cp_data=get_cp_data(cp_list,smallest_r,str_df,independent_cp)
        #遍历cp_data找出所有的独立结构
        find_str=find_independent_str(smiles,smallest_r,cp_data,independent_cp,dependent_cp)
        string0=find_str[0]
        index_data=find_str[1]
        index_cp=find_str[2]
        index_data0=find_str[3]
        inner_dist=find_str[4]
        inner_dist,end_atom_num,end_index=get_atom_num(smiles,inner_dist)
        end_atom_pair={'pair_atom':end_atom_num,'pair_index':end_index}
        total_end_atom_pair[name]=end_atom_pair
        total_inner_dist[name]=inner_dist
        #对独立结构进行细节调整
        index_data2={}
        for k,v in index_data.items():
            j=v[1]
            j2=add_bracket(j)
            j3=make_smi(j2)
            j4=link_c(j3)
            mol=Chem.MolFromSmiles(j4)
            if mol:
                 j6= Chem.MolToSmiles(mol)
            v2=[v[0],j6]
            index_data2[k]=v2
        make_con_data=make_con(index_data2,index_cp)
        index_data3=make_con_data[0]
        index_data4=delete_free_radical_in_index_data(index_data3)
        index_cp2=make_con_data[1]
        for k,v in index_data4.items():
            
            ring_total_list.append(v[1])
            
        neighbor_data=found_neighbor(str_df,index_data3,index_cp2)
        neighbor_data2=found_end_point_neighbour(smiles,neighbor_data,index_data4) 
    #去除'[C]'
        for k,v in neighbor_data2.items():
            for k2,v2 in v['right_neighbor'].items():
                if '[C]'in v2:
                    v2=v2.replace('[C]','C')
            for k2,v2 in v['left_neighbor'].items():
                if '[C]'in v2:
                    v2=v2.replace('[C]','C')
            if '[C]'in v['self']:
                v['self']=v['self'].replace('[C]','C')
        total_neighbor_data[name]=neighbor_data2
        n=n+1#总循环迭代数
    ring_total_list=sorted(list(set(ring_total_list)))
    total_inner_dist2={}
    for key,values in total_inner_dist.items():
        if key in name_list:
            total_inner_dist2[key]=values
    return ring_total_list,total_neighbor_data,total_inner_dist2,total_end_atom_pair

def find_independent_str(smiles,smallest_r0,cp_data0,independent_cp0,dependent_cp0):
    independent_ring_string0=[]#存储形成独立环的SMILES码
    index_data={}
    index_data0={}
    index_cp=[]
    inner_dist = {}
    for i in smallest_r0:
        if i in independent_cp0:
            name=str(i[0])+"-"+str(i[1])
            p=smiles[i[0]:i[1]+1]
            if p!="([C])":
                independent_ring_string0.append(p)
                index_data[name]=[i,p]
                index_data0[name]=[i,p]
                index_cp.append(i)
                inner_dist[name]=[i[0],i[1]]
    for k,v in cp_data0.items():
        independent_num=[]
        inner_num=[]
        w=v[1:]
        first=v[0][0]
        last=v[0][1]
        name=str(first)+"-"+str(last)
        if v[0] in independent_cp0:
            inner_num=[first,last]
            for i in w:
                if i in independent_cp0:
                    inner_num.append(i[0])
                    inner_num.append(i[1])
                    independent_num.append(i[0])
                    independent_num.append(i[1])
                #if i in bratch_cp0:
                #    inner_num.append(i[0])
                #    inner_num.append(i[1])
                #    independent_num.append(i[0])
                #    independent_num.append(i[1])
                if i in dependent_cp0:
                    if i not in smallest_r0:
                        inner_name=str(i[0])+"-"+str(i[1])
                        j=cp_data[inner_name]
                        for k in j[1:]:
                            if k in independent_cp0:
                                inner_num.append(k[0])
                                inner_num.append(k[1])
                                independent_num.append(k[0])
                                independent_num.append(k[1])
                            #if k in bratch_cp0:
                            #    inner_num.append(k[0])
                            #    inner_num.append(k[1])
                            #    independent_num.append(k[0])
                            #    independent_num.append(k[1])
            inner_dist[name]=inner_num
        if len(inner_num)>=1:
            inner_num.sort()
            slice_s=""
            i=0
            while i < (len(inner_num)-1):
                j=inner_num[i]
                k=inner_num[i+1]
                s1=smiles[j:k+1]
                slice_s=slice_s+s1
                if i <(len(inner_num)-2):
                    l=inner_num[i+2]
                    s2='C'
                    slice_s=slice_s+s2
                i=i+2
            index_data0[name]=[inner_num,slice_s]
            independent_ring_string0.append(slice_s)
            index_data[name]=[v[0],slice_s]
            index_cp.append(v[0])
    if len(independent_ring_string0)==0:
        independent_ring_string0.append(smiles)
    return independent_ring_string0,index_data,index_cp,index_data0,inner_dist

def get_cp_data(cp_list0,smallest_r0,str_df0,independent_cp):
    cp_data0={}#用于存放非最小括号的内一层
    cp_name_list=[]#用于存放非最小括号的名称
    for cp in cp_list0:
        if cp not in smallest_r0:
            cp_name=str(cp[0])+"-"+str(cp[1])
            cp_name_list.append(cp_name)
            cp_data0[cp_name]=[cp]
    #找到每个括号的所有内一层括号        
    hang=len(smallest_r0)-1
    while hang >=0:
        lie=len(cp_list0)-1
        while lie >0:
            cp=str_df0[lie][hang]
            if len(cp) == 2:
                cp_name=str(cp[0])+"-"+str(cp[1])
                cp=str_df0[lie][hang]
                cp_min=str_df0[lie-1][hang]
                j=lie-1
                while (cp_min not in independent_cp) and (j!=0):
                    j=j-1
                    cp_min=str_df0[j][hang]
                if cp_min in independent_cp:
                    for k,v in cp_data0.items():
                        if k==cp_name:
                            if cp_min not in v:
                                v.append(cp_min)
            lie=lie-1
        hang=hang-1
    return(cp_data0)



def check_result(name_list,total_neighbor_data,total_inner_dist):
    #用于检查是否有
    for idx,i in enumerate(name_list):
        print(i)
        smiles = smi_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        fig = Draw.MolToImage(mol, size=(500,500), kekulize=True)
        display(fig)
        neighbor_data = total_neighbor_data[i]
        #for k,v in neighbor_data.items():
            #pu = v['self']
            #pu_mol = Chem.MolFromSmiles(pu)
            #pu_fig = Draw.MolToImage(pu_mol, size=(100,100), kekulize=True)
            #display(pu_fig)
        inner_dist = total_inner_dist[i]
        for k,v in inner_dist.items():
            hight_atom(mol,v)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
def hight_atom(mol,highlight_atoms):
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel',str(atom.GetIdx()))
    image_data = BytesIO()
    view = rdMolDraw2D.MolDraw2DSVG(1000, 1000)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    option = view.drawOptions()
    for atom in mol.GetAtoms():
        option.atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx() + 1)
    view.DrawMolecule(tm, highlightAtoms=highlight_atoms)
    view.FinishDrawing()
    svg = view.GetDrawingText()
    SVG(svg.replace('svg:', ''))
    svg2png(bytestring=svg, write_to='test.png')
    img = Image.open('test.png')
    #img.save(image_data, format='PNG')
    display(img)
    
    
def list_to_array(x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(x)],axis=1)
    
    return dff.fillna(-10).values.T.astype(int)

def dist_to_dataframe(dist):
    inner_list = list(inner_dist.values())
    name_index = list(inner_dist.keys())
    inner_array = list_to_array(inner_list)
    inner_df = pd.DataFrame(inner_array,index = name_index)
    
    return inner_df



def get_bratch_dist2(smi_list,name_list):
    total_bratch_dist = {}
    link_symbol_to_delete = [f"[*:{i}]" for i in range(20)]
    for idx,smi in enumerate(smi_list):
        name = name_list[idx]
        bratch_list = []
        inner_list = []
        bratch_dist = {}
        core_dist ={}
        m = Chem.MolFromSmiles(smi)
        #for atom in m.GetAtoms():
            #atom.SetProp('atomLabel',str(atom.GetIdx()))
        
        #显示分子结构
        #fig = Draw.MolToImage(m, size=(800,800), kekulize=True)
        #display(fig)
        core = MurckoScaffold.GetScaffoldForMol(m)
        core_smi = Chem.MolToSmiles(core)
        flag =m.HasSubstructMatch(core)
        if flag:
            C_atomids = m.GetSubstructMatches(core)
            C_atomids = list(C_atomids[0])
        core_dist['smiles'] = core_smi
        core_dist['inner'] = C_atomids
        bratch_dist['core'] = core_dist
        m_core = [m, core]
        res, unmatched = rdRGD.RGroupDecompose([core], [m], asSmiles=True)
        v_list = []
        for k,v in res[0].items():
            if k != 'Core':
                for i in link_symbol_to_delete:
                    if i in v:
                        v=v.replace(i,'')
                if '.' in v:
                    cut_index = v.index('.')
                    v1 = v[cut_index+1:]
                    v2 = v[:cut_index]
                    v_list.append(v1)
                    v_list.append(v2)
                else:
                    v_list.append(v)
        for v in v_list:
            #add_bratch_to_list(v,bratch_list)
            br_m = Chem.MolFromSmiles(v)
            if br_m:
                num = br_m.GetNumAtoms()
                if num > 1:
                    bratch_list.append(v)
            else:
                print("mol error")
                
        bratch_list = list(set(bratch_list)) 
        atomids_set = []

        for idx,i in enumerate(bratch_list):
            br_dist ={'smiles':i,'inner':[]}
            br_m = Chem.MolFromSmiles(i)
            flag =m.HasSubstructMatch(br_m)
            if flag:
                atomids_list = []
                atomids = m.GetSubstructMatches(br_m)
                for n in atomids:
                    not_bratch = list(set(n) & set(C_atomids))
                    not_br_num = len(not_bratch)
                    if not_br_num == 0:
                        atom_num = len(n)
                        atomids_list.append(n)

                for j in atomids_list:
                    same = 0
                    for k in atomids_list:
                        if len(list(set(k) & set(j))) == 0:
                            same = same + 1
                    if same == (len(atomids_list)-1):
                        atomids_set.append(j)
                        br_dist['inner'].append(list(j))
            else:
                print("No branch chain matches")
                #print(i)
                #display(br_m)
                #fig = Draw.MolToImage(m, size=(1000,1000), kekulize=True)
                #display(fig)
                #print(name)
            bratch_dist[idx]=br_dist
        delete_inner_list = []
        for i in atomids_set:
            same = 0
            for j in atomids_set:
                if len(list(set(i) & set(j))) != 0:
                    if len(i)>len(j):
                        delete_inner_list.append(list(j))
                    if len(i)<len(j):
                        delete_inner_list.append(list(i))

        for k,v in bratch_dist.items():
            if k != 'core':
                inner_list = v['inner']
                for i in inner_list:
                    if i in delete_inner_list:
                        inner_list.remove(i)
                v['inner'] = inner_list
                
        #将一些过小的支链连接到基元上面
        atom_num =  m.GetNumAtoms()
        atom_list = []
        for k,v in bratch_dist.items():
            inner = v['inner']
            if k == 'core':
                atom_list = atom_list+inner
            else:
                for i in inner:
                    atom_list = atom_list+i

        atom_list = list(set(atom_list))
        res_list = list(set(list(range(atom_num))).difference(set(atom_list))) 
        res_list_minus = [i-1 for i in res_list]
        res_list_plus = [i+1 for i in res_list]   
        inner = bratch_dist['core']['inner']
        inner2 = inner.copy()
        for i in inner:
            if i in res_list_minus:
                inner2.append(i+1)
            if i in res_list_plus:
                inner2.append(i-1)
        inner2 = list(set(inner2))
        inner2.sort()
        bratch_dist['core']['inner'] = inner2
        
        #根据补充进来的原子更新SMILES
        new_smiles = AllChem.MolFragmentToSmiles(m,bratch_dist['core']['inner'])
        new_mol=Chem.MolFromSmiles(new_smiles)
        if new_mol:
            bratch_dist['core']['smiles']=new_smiles
        #else:
        #    print("ERROR")
        #    print(new_smiles)
        #    print(bratch_dist['core']['smiles'])
        total_bratch_dist[name]=bratch_dist    
    return total_bratch_dist

def update_bratch0(name_list,smi_list,total_neighbor_data,total_inner_dist,total_bratch_dist):
    ring_total_list = []
    for idx,name in enumerate(name_list):
        
        smiles = smi_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        neighbor_data = total_neighbor_data[name]
        inner_dist = total_inner_dist[name]
        bratch_dist= total_bratch_dist[name]
        core_atom_index = bratch_dist['core']['inner']
        inner_list = list(inner_dist.values())
        name_index = list(inner_dist.keys())
        
        try:
            inner_array = list_to_array(inner_list)
        except ValueError:
            print("name")
            print(name)
            print("inner_list")
            print(inner_list)
            continue
        #删除基元中的支链部分
        error_key = []
        for k,v in inner_dist.items():
            v_ = list(set(v).intersection(set(core_atom_index)))
            if len(v_) != 0:
                fragsmi= AllChem.MolFragmentToSmiles(mol,v_)
                fragmol = Chem.MolFromSmiles(fragsmi)
                inner_dist[k]=v_
                if fragmol:
                    if '[C]' in fragsmi:
                        fragsmi=fragsmi.replace('[C]','')
                    neighbor_data[k]['self']=fragsmi
        for k in error_key:
            del inner_dist[k]
            del neighbor_data[k]
            
        #将支链添加至总体性质
        for k,v in bratch_dist.items():
            if k != 'core':
                inner_list_set = v['inner']
                br_smiles = v['smiles']
                for i in inner_list_set:

                    link_atom = min(i)
                    pair_num = link_atom-1
                    locat = np.argwhere(inner_array==pair_num)
                    if len(locat)== 1:
                        
                        neighbor_node_name = name_index[locat[0][0]]
                        node_name = neighbor_node_name + '-' + str(pair_num)
                        neighbor_smiles = neighbor_data[neighbor_node_name]['self']
                        neighbor_data[neighbor_node_name]['left_neighbor'][node_name]=br_smiles
                        update_dist = {node_name:{'self':br_smiles,'right_neighbor':{neighbor_node_name:neighbor_smiles},'left_neighbor':{}}}
                        neighbor_data.update(update_dist)
                        inner_dist[node_name]=i
                    if len(locat)==0:
                        link_atom = max(i)
                        pair_num = link_atom+1
                        locat = np.argwhere(inner_array==pair_num)
                        if len(locat)== 1:
                            neighbor_node_name = name_index[locat[0][0]]
                            node_name = neighbor_node_name + '-' + str(pair_num)
                            neighbor_smiles = neighbor_data[neighbor_node_name]['self']
                            neighbor_data[neighbor_node_name]['left_neighbor'][node_name]=br_smiles
                            update_dist = {node_name:{'self':br_smiles,'right_neighbor':{neighbor_node_name:neighbor_smiles},'left_neighbor':{}}}
                            neighbor_data.update(update_dist)
                            inner_dist[node_name]=i
                       
        for k,v in neighbor_data.items():
            
            right_neighbor = v['right_neighbor']
            for k2,v2 in right_neighbor.items():
                neighbor_data[k]['right_neighbor'][k2]=neighbor_data[k2]['self']
            left_neighbor = v['left_neighbor']
            for k2,v2 in left_neighbor.items():
                neighbor_data[k]['left_neighbor'][k2]=neighbor_data[k2]['self']
        #检查结果
        #for k,v in inner_dist.items():
            #hight_atom(mol,v)
        #print('***************************************************************')
        
        total_neighbor_data[name]=neighbor_data
        total_inner_dist[name]=inner_dist
        for k,v in neighbor_data.items():
            ring_total_list.append(v['self'])
    return total_neighbor_data,total_inner_dist,ring_total_list

def update_bratch(name_list,smi_list,total_neighbor_data,total_inner_dist,total_bratch_dist):
    ring_total_list = []
    for idx,name in enumerate(name_list):
        smiles = smi_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        neighbor_data = total_neighbor_data[name]
        inner_dist = total_inner_dist[name]
        bratch_dist= total_bratch_dist[name]
        core_atom_index = bratch_dist['core']['inner']
        inner_list = list(inner_dist.values())
        name_index = list(inner_dist.keys())
        
        try:
            inner_array = list_to_array(inner_list)
        except ValueError:
            print("name")
            print(name)
            print("inner_list")
            print(inner_list)
            continue
        #删除基元中的支链部分
        error_key = []
        for k,v in inner_dist.items():
            v_ = list(set(v).intersection(set(core_atom_index)))
            if len(v_) != 0:
                fragsmi= AllChem.MolFragmentToSmiles(mol,v_)
                if '[C]' in fragsmi:
                    fragsmi=fragsmi.replace('[C]','')
                fragmol = Chem.MolFromSmiles(fragsmi)
                inner_dist[k]=v_
                if fragmol:
                    fragsmi = Chem.MolToSmiles(fragmol)
                    neighbor_data[k]['self']=fragsmi
                else:
                    #由于氮的不饱和作用，为了能使SMILES完整替，将一些'n'替换成了'[nH]',所以inner_dist可能和neighbor_data不能严格对上
                    if 'nnn' in fragsmi:
                        fragsmi2 = fragsmi.replace('nnn','n[nH]n')
                        fragmol = Chem.MolFromSmiles(fragsmi2)
                        if fragmol:
                            fragsmi = Chem.MolToSmiles(fragmol)
                            neighbor_data[k]['self']=fragsmi
                    elif 'n' in fragsmi:
                        indices = [index for index, element in enumerate(fragsmi) if element == 'n']
                        if_mol=False
                        smilist = list(fragsmi)
                        for i in indices:
                            smi2 = smilist.copy()
                            smi2[i]='[nH]'
                            a = ''
                            smi3 = a.join(smi2) 
                            fragmol = Chem.MolFromSmiles(smi3)
                            if fragmol:
                                if_mol = True
                                break
                        if if_mol:
                            fragsmi = Chem.MolToSmiles(fragmol)
                            neighbor_data[k]['self']=fragsmi
                        else:
                            smi2 = fragsmi.replace('n','[nH]')
                            fragmol = Chem.MolFromSmiles(smi2)
                            if fragmol:
                                fragsmi = Chem.MolToSmiles(fragmol)
                                neighbor_data[k]['self']=fragsmi
                            else:
                                print('2')
                                print(smi)
                    else:
                        neighbor_data[k]['self']=fragsmi
                       
        for k in error_key:
            del inner_dist[k]
            del neighbor_data[k]
            
        #将支链添加至总体性质
        for k,v in bratch_dist.items():
            if k != 'core':
                inner_list_set = v['inner']
                br_smiles = v['smiles']
                for i in inner_list_set:

                    link_atom = min(i)
                    pair_num = link_atom-1
                    locat = np.argwhere(inner_array==pair_num)
                    if len(locat)== 1:
                        
                        neighbor_node_name = name_index[locat[0][0]]
                        node_name = neighbor_node_name + '-' + str(pair_num)
                        neighbor_smiles = neighbor_data[neighbor_node_name]['self']
                        neighbor_data[neighbor_node_name]['left_neighbor'][node_name]=br_smiles
                        update_dist = {node_name:{'self':br_smiles,'right_neighbor':{neighbor_node_name:neighbor_smiles},'left_neighbor':{}}}
                        neighbor_data.update(update_dist)
                        inner_dist[node_name]=i
                    if len(locat)==0:
                        link_atom = max(i)
                        pair_num = link_atom+1
                        locat = np.argwhere(inner_array==pair_num)
                        if len(locat)== 1:
                            neighbor_node_name = name_index[locat[0][0]]
                            node_name = neighbor_node_name + '-' + str(pair_num)
                            neighbor_smiles = neighbor_data[neighbor_node_name]['self']
                            neighbor_data[neighbor_node_name]['left_neighbor'][node_name]=br_smiles
                            update_dist = {node_name:{'self':br_smiles,'right_neighbor':{neighbor_node_name:neighbor_smiles},'left_neighbor':{}}}
                            neighbor_data.update(update_dist)
                            inner_dist[node_name]=i
                       
        for k,v in neighbor_data.items():
            
            right_neighbor = v['right_neighbor']
            for k2,v2 in right_neighbor.items():
                neighbor_data[k]['right_neighbor'][k2]=neighbor_data[k2]['self']
            left_neighbor = v['left_neighbor']
            for k2,v2 in left_neighbor.items():
                neighbor_data[k]['left_neighbor'][k2]=neighbor_data[k2]['self']
        #检查结果
        #for k,v in inner_dist.items():
            #hight_atom(mol,v)
        #print('***************************************************************')
        
        total_neighbor_data[name]=neighbor_data
        total_inner_dist[name]=inner_dist
        for k,v in neighbor_data.items():
            ring_total_list.append(v['self'])
        ring_total_list = list(set(ring_total_list))
    return total_neighbor_data,total_inner_dist,ring_total_list

def cut_pu(smiles):
    
    from_get_bracket_index=get_bracket_index(smiles)
    left_index_list=from_get_bracket_index[0]#存储“（”的索引
    right_index_list=from_get_bracket_index[1]#存储“)”的索引
    index_list=from_get_bracket_index[2]#存储所有括号的索引
    ##对括号进行匹配
    cp_list=pairing(smiles,index_list,left_index_list,right_index_list)
    cp_arr=np.array(cp_list)#将储存括号对的列表转化为np.array
    index_arr=np.array(index_list)#将存储所有括号索引的列表转化为np.array

    #识别最小括号
    smallest_r=smallest(cp_list,index_arr)
    
    #以最小独立区域为起点，DataFrame。第一列是最小独立区域，第二列是外一层，第三列是外二层
    str_df=structure_DataFrame(cp_list,smallest_r,right_index_list,left_index_list)

    #创建三个列表，将独立括号区域进行分类，分成独立环，支链，非独立环三种
    independent_cp_and_dependent_cp=rigin_type_classify(cp_list,smiles,smallest_r,str_df)
    independent_cp=independent_cp_and_dependent_cp[0]
    dependent_cp=independent_cp_and_dependent_cp[1]
    cp_data=get_cp_data(cp_list,smallest_r,str_df,independent_cp)
    #遍历cp_data找出所有的独立结构
    independent_ring_string0,index_data,index_cp,index_data0,inner_dist=find_independent_str(smiles,smallest_r,cp_data,independent_cp,dependent_cp)
    if len(independent_ring_string0) > 1:
        independent_ring_string=[]
        for i in independent_ring_string0:
            if i[0]=='(' and i[-1]==')':
                i = i[1:]
                i = i[:-1]
            if i[0]=='-':
                i = i[1:]
            if i[0]=='=':
                i = i[1:]
            if i[0]=='#':
                i=i[1:]
            independent_ring_string.append(i)
        independent_ring_string = list(set(independent_ring_string))
        return independent_ring_string
    else:
        return [smiles]
    
def get_new_neighbor_data(total_neighbor_data,total_inner_dist,name_list,smi_list):
    #step 0 识别出来没有正确划分的基元并标记出对应的编号
    replace_neighbor_data = {}
    total_inner_dist2 = {}
    ring_total_list2 = []
    for idx,name in enumerate(name_list):
        smiles = smi_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        inner_dist =  total_inner_dist[name]
        neighbor_data = total_neighbor_data[name]
        new_inner = {}
        replace_nei = {}
        for k,v in neighbor_data.items():
            replace ={}
            pu = v['self']
            right = list(v['right_neighbor'].keys())
            left = list(v['left_neighbor'].keys())
            pu_mol = Chem.MolFromSmiles(pu)
            inner = inner_dist[k]
            ring_list = cut_pu(pu)
            if len(ring_list)==1:
                new_inner[k]=inner
            else:
                for sdx,smi in enumerate(ring_list):
                    new_key = k + '-cut-' + str(sdx)
                    m = Chem.MolFromSmiles(smi)
                    flag =m.HasSubstructMatch(m)
                    if flag:
                        atomids = mol.GetSubstructMatches(m)
                        if len(atomids)==0:
                            new_inner[k]=inner
                            break
                        else:
                            for ids in atomids:
                                ids = list(ids)
                                intersection = list(set(ids) & set(inner))
                                if len(intersection)==len(ids):
                                    new_inner[new_key]=ids
                                    smi = Chem.MolToSmiles(m)
                                    replace[new_key]=smi
                    else:
                        print("no flag")
                        new_inner[k]=inner
            
            if len(replace)> 1:
                replace_nei[k]=replace
            else:
                replace_nei[k]={}
        total_inner_dist2[name]=new_inner
        replace_neighbor_data[name]=replace_nei
        
    #step1:将新拆分出的基元加入total_neighbor_data
    total_neighbor_data2 = {}
    total_delete_key = {}
    total_add_key = {}
    for key,neighbor_data in total_neighbor_data.items():
        replace_data = replace_neighbor_data[key]
        new_neighbor_data = {}
        delete_key = []
        add_key = []
        for k,replace in replace_data.items():
            neighbor = neighbor_data[k]
            if len(replace)==0:
                new_neighbor_data[k]=neighbor_data[k]
            else:
                delete_key.append(k)
                for k2,v2 in replace.items():
                    add_key.append(k2)
                    new_neighbor = {'self':v2,'right_neighbor':{},'left_neighbor':{}}
                    new_neighbor_data[k2]=new_neighbor
        total_neighbor_data2[key]=new_neighbor_data
        total_delete_key[key]=delete_key
        total_add_key[key]=add_key 
        
    #step2:去除right_neighbor及left_neighbor里面的原未拆分基元
    total_neighbor_data3={}
    total_add_key2={}
    for key,neighbor_data in total_neighbor_data2.items():
        delete_key=total_delete_key[key]
        new_neighbor_data = {}
        add_key=total_add_key[key]
        for k,v in neighbor_data.items():
            neighbor = {}
            v2=v.copy()
            self = v2['self']
            right = v2['right_neighbor']
            left = v2['left_neighbor']        
            right_del = set(list(v2['right_neighbor'].keys())) & set(delete_key)
            left_del = set(list(v2['left_neighbor'].keys())) & set(delete_key)
            if len(right_del) != 0:
                add_key.append(k)
            elif len(left_del) != 0:
                add_key.append(k)
            for i in right_del:
                del right[i]
            for i in left_del:
                del left[i]
            neighbor['self']=self
            neighbor['right_neighbor']=right
            neighbor['left_neighbor']=left
            new_neighbor_data[k]=neighbor
        total_neighbor_data3[key]=new_neighbor_data
        total_add_key2[key]=add_key
        
    #step 3:建立新的连接关系
    total_neighbor_data4 = {}
    for idx,name in enumerate(name_list):
        smiles = smi_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        add_key = total_add_key2[name]
        inner_dist = total_inner_dist2[name]
        neighbor_data = total_neighbor_data3[name]
        new_neighbor = []
        for i in add_key:
            inner = inner_dist[i]
            add_key2 = add_key.copy()
            add_key2.remove(i)
            for j in add_key2:
                inner2 = inner_dist[j]
                for k in inner:
                    for l in inner2:
                        bond = mol.GetBondBetweenAtoms(k,l)
                        if bond:
                            pair = [i,j]
                            pair.sort()
                            if pair not in new_neighbor:
                                new_neighbor.append(pair)
        new_neighbor_data = neighbor_data.copy()
        for i in new_neighbor:
            ki = i[0]
            r_nei = neighbor_data[i[1]]['self']
            new_neighbor_data[ki]['right_neighbor'][i[1]]=r_nei
            
        total_neighbor_data4[name]=new_neighbor_data 
    #整理返回参数    
    for k,v in total_neighbor_data4.items():
        for k2,v2 in v.items():
            ring_total_list2.append(v2['self'])
    ring_total_list2 = list(set(ring_total_list2))
    ring_total_list2.sort()
    return total_neighbor_data4,total_inner_dist2,ring_total_list2


# In[38]:


def main(file_name):
    print("将csv文件里面的内容整理成列表...")
    smi_list,name_list0,name_list,mol_list,num_list=process_smiles(file_name)
    print("将SMILES码里面的基元识别整理出来...")
    ring_total_list0,total_neighbor_data0,total_inner_dist0,total_end_atom_pair=get_pu(smi_list,name_list)
    print("处理支链性质...")
    total_bratch_dist = get_bratch_dist2(smi_list,name_list)
    print("将支链信息与碳链骨架信息合并...")
    total_neighbor_data,total_inner_dist,ring_total_list = update_bratch(name_list,smi_list,total_neighbor_data0,total_inner_dist0,total_bratch_dist)
    total_neighbor_data2,total_inner_dist2,ring_total_list2 = get_new_neighbor_data(total_neighbor_data,total_inner_dist,name_list,smi_list)
    print("生成连接矩阵")
    matrix_list=get_adj(ring_total_list2,total_neighbor_data2,name_list)
    print("每个节点对应的pu种类")
    pu_index = get_pu_dict(total_neighbor_data2,ring_total_list2)
    print("节点之间连接原子的索引")
    pair_atom_dist = get_pair_atom(total_neighbor_data2,total_inner_dist2)
    print("生成每个polymer-unit的MACCS码")
    MACCS_dict = get_MACCS(ring_total_list2,False)
    print("生成边索引，边个数列表")
    senders_list,receivers_list,edge_num_list = edge_index(matrix_list)
    print("生成节点索引，节点个数列表")
    node_index, node_num_list = get_node_index(total_neighbor_data2,ring_total_list2)
    print("生成边特征值")
    edge_list = get_edge_feature(node_num_list,mol_list,name_list,matrix_list,pu_index,pair_atom_dist,total_inner_dist)


# In[ ]:


if __name__ == '__main__':
    main('polymer.csv')

