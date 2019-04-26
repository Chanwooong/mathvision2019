%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Face Recognition Using PCA by KCW 20190422   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
%% parmeter setting
M = 360;   %training data number : class 40, class per picture 9 -> 40x9 
w = 46;    %width
h = 56;    %hight
k_num = 100;  %number of PCA vector
DB_path = cd; % current Direcotry
test_idx = [1:40]; %test input data index
result_table = zeros(40, 2); 
%% training data setting
X=[];
for i=1:40
    for j= 2 :10
    filename = sprintf('s%d_%d.png', i,j);
    filename = [DB_path '\Training\' filename];
    img=imread(filename);    
    tmp=reshape(img', h*w,1);   
    X=[X tmp];             
    end
end

% total training data noralization
m = 0;
st = 0;
for i=1:M
    m=m+mean(double(X(:,i)));
    st=st+std(double(X(:,i)));
end
total_m = m/M;
total_std = st/M;
for i=1:M
    temp=double(X(:,i));
    m=mean(temp);
    st=std(temp);
    X(:,i)=(temp-m)*total_std/st+total_m;
end

for i=1:40
    for j =2 :10
    filename = sprintf('s%d_%d.png', i,j);
    filename = [DB_path '\Normalization\' filename];
    img = reshape(X(:,j+9*(i-1)-1), w, h);
    img = img';
    imwrite(img, filename);         
    end
end

%% mean face 
m_face=mean(X,2);           
tmimg=uint8(m_face);         
img=reshape(tmimg,w,h); 
img=img';  
imwrite(img, [cd '\mean_face.png']);

Input_dummy=zeros(w*h,M);
for i=1:M
    Input_dummy(:,i)=double(X(:,i));
end

 
%% do it! PCA
A=Input_dummy;                    
L=A'*A;                   
[vv, dd]=eig(L);           
v = zeros(360, k_num);
d = zeros(1,k_num);
for k=1:k_num
    d(1,k)= dd(360-k+1,360-k+1);
    v(:,k) = vv(:, 360-k+1);
end

%% eigen vector normalize

for i=1:k_num
    v_norm=v(:,i);
    tmp=sqrt(sum(v_norm.^2));
    v(:,i)=v(:,i)./tmp;
end

%% KL and normalize
u = Input_dummy*v;
for i=1:k_num
    tmp=u(:,i);
    tmp=sqrt(sum(tmp.^2));
    u(:,i)=u(:,i)./tmp;
end
training_W = u'*Input_dummy;  %training data coff 

%% eigen face
for i=1:k_num
    img=reshape(u(:,i),w,h);
    img=img';
    img=histeq(img,255); %%% histogram equalization
    filename = sprintf('%d_eigenface_%d.png',k_num, i);
    filename = [DB_path '\eigenface\' filename];
    imwrite(img, filename);         
end

%% test part
l=0; % count
for idx=1:40
    filename = sprintf('s%d_1.png',idx);
    filename  = [DB_path '\Test\' filename];
  %  filename = 'ref_kcw_test.png'; %%% for kcw test
    InputImage = imread(filename);
    InImage=reshape(double(InputImage)',h*w,1);
    temp=InImage;
    me=mean(temp);
    st=std(temp);
    temp=(temp-me)*total_std/st+total_m;
    NormImage= temp;
    Diff = NormImage-m_face;

    %------------Reconstruct face!!!-------------
    %%%%%  reconstruct = meanface + (eigenface * new weight)
    p = u'*NormImage;   
    Recon = m_face+u*p;   %  m + (e_f*n_w)
    max_v = max(Recon);
    Recon = Recon/max_v;
    Recon = reshape(Recon,w,h);
    Recon = Recon';
    filename = sprintf('s%d_k%d.png',idx, k_num);
    filename = [DB_path '\Reconstruction\' filename];
    imwrite(Recon,filename);
    %---------------------------------------

    test_W = u'*Diff;  %test data coff 

    %%take Euclidean distance
    e = zeros(1,M);
    test_W_M = repmat(test_W,1,M);
    Eu_d = test_W_M - training_W;
    for i=1:M
        e(1,i) = norm(Eu_d(:,i));
    end
    tmp = zeros(1,40);
    for i=1:40
        tmp(1,i) =  sum(e(1,9*(i-1)+1:9*(i))); 
    end
    tmp_max = max(tmp);
    for i=1:40
        tmp(1,i) =  tmp_max - tmp(1,i); 
    end
    %bar(tmp) %% plot result
    result_table(idx,1)=idx;
    [tmp, max_index] = max(tmp);
    result_table(idx,2) = max_index;
    if (idx == max_index)
        l= l+1;
    end
end
filename = sprintf('k_num%d_result.txt',k_num);
dlmwrite(filename, result_table);
Recong_rate = l/40*100






