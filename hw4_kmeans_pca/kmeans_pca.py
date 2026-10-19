import numpy as np
import matplotlib.pyplot as plt


kernel=np.array([[10,0.5],[-10,0.25]])
kernel=np.linalg.inv(kernel.T @ kernel)
    

def calDist(A,B):
    if(len(A)<2 or len(B)<2):
        print("expect longer than 2")

    #ans=float(A[0]-B[0])**2/49+float(A[1]-B[1])**2
    ans=float(A[0]-B[0])**2+float(A[1]-B[1])**2
    #ans=float(max(A[0]-B[0],A[1]-B[1]))
    #ans=float(A[0]-B[0]+A[1]-B[1])
    #ans=float(A[1]-B[1])
    #print(ans)

    #ans=(A[:-1]-B) @ kernel @ (A[:-1]-B).T 
    #special dist mamhalton
    
    return ans

def kmean(dataPoint,X):
    #X=np.array([[0,-8],[0,-5],[0,-1],[0,1],[0,5]],dtype='float')
    
    u=[-100,-100]
    l=[100,100]
    for i in dataPoint:
        if(i[0]>u[0]):
            u[0]=i[0]
        if(i[0]<l[0]):
            l[0]=i[0]
        if(i[1]>u[1]):
            u[1]=i[1]
        if(i[1]<l[1]):
            l[1]=i[1]
        i.append(-1)
    scale=[(u[0]-l[0]),u[1]-l[1]]
    dataPoint=np.array(dataPoint,dtype='float')
    
    #plt.scatter(dataPoint[:,0],dataPoint[:,1],c='red')
    #plt.show()
    

    err=100
    tmp=0
    mini=0
    idx=0
    sumi=np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype='float')
    x=0
    y=0
    step=0
    while(err>0.000000001 and step<100):
        print(step)
        step+=1
        err=0
        for i in dataPoint:
            mini = calDist(i,X[0])
            idx=0
            for j in range(1,5):
                tmp=calDist(i,X[j])
                if(tmp<mini):
                    mini=tmp
                    idx=j
            sumi[idx][2]+=1
            sumi[idx][0]+=i[0]
            sumi[idx][1]+=i[1]
            i[2]=idx
        for i in range(5):
            if(sumi[i][2]==0):
                continue
            x=sumi[i][0]/sumi[i][2]
            y=sumi[i][1]/sumi[i][2]
            err=max(calDist(X[i],[x,y]),err)
            X[i]=[x,y]
            sumi[i]=[0,0,0]

    result=[[],[],[],[],[]]
    
    for i in dataPoint:
        result[int(i[2])].append([i[0],i[1]])
    color=['blue','black','yellow','green','purple']
    for i in range(5):
        
        if(len(result[i])==0):
            continue;
        plt.scatter(np.array(result[i])[:,0],np.array(result[i])[:,1],c=color[i])
    
    plt.scatter(X[:,0],X[:,1],c="red")
    return result

def PCA(data,num=0):
    data=np.array(data,dtype='float')

    
    data=np.array(data,dtype='float')
    cov=np.zeros((2,2))
    n=len(data)
    mean_x=sum(data[:,0])/n
    mean_y=sum(data[:,1])/n
    cov[0][0]=sum((data[:,0]-mean_x)**2)/(n-1)
    cov[1][1]=sum((data[:,1]-mean_y)**2)/(n-1)
    cov[0][1]=sum((data[:,0]-mean_x)*(data[:,1]-mean_y))/(n-1)
    cov[1][0]=cov[0][1]
    eigval,eigvec=np.linalg.eig(cov)
    eigorder=np.argsort(eigval)[::-1]
    
    #return (eigvec.T @ data.T).T[eigorder[0]]
    return eigvec[:,eigorder[0]],[mean_x,mean_y]

if __name__ =="__main__":
    fileName='data'    
    data=[]
    with open(fileName,'r') as file:
        for line in file:
            data.append([float(item.rstrip()) for item in line.split(',') if item !='\n'])
    data=data[:-1]
    
    X=np.array([[10,10],[-10,-10],[2,2],[3,3],[-3,-3]],dtype='float')
    
    #print(len(data[0]))
    import copy
    data_=copy.deepcopy(data)
    result=kmean(data_,X)
    #print(len(data[0]))
    pEigvec,meanPos=PCA(data)
    print(pEigvec)

    #print(len(data[0]))
    pEigvecs=[[],[],[],[],[]]
    meanPoss=[[],[],[],[],[]]
    color=['blue','black','yellow','green','purple']

    for i in range(len(result)):
        pEigvecs[i],meanPoss[i]=PCA(result[i])
    print(pEigvecs)
    plt.title('the result of kmean and PCA',c='black')
    plt.show();