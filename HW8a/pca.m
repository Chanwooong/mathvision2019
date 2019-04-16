clear all;
a = load('data_a.txt');
b = load('data_b.txt');
[sz_a_x, sz_a_y] = size(a);
[sz_b_x, sz_b_y] = size(b);
x = [a;b]; %%% a,b concatenate
test = load('test.txt');
%% pca
c = cov(x);
[v,d] = eig(c); %%main vector 4, 3
eig_v = zeros(sz_a_y,1);
for i=1:sz_a_y   %% find ref eig_value and index
    eig_v(i,1) = d(i,i);
end
[out, idx] = sort(eig_v);
c_fir = x*v(:,idx(sz_a_y)); 
c_sec = x*v(:,idx(sz_a_y-1));

pca_a = [c_fir(1:sz_a_x), c_sec(1:sz_a_x)];
pca_b = [c_fir(sz_a_x+1:sz_a_x+sz_b_x), c_sec(sz_a_x+1:sz_a_x+sz_b_x)];
%% plot(S:coordinate)
figure(1);
plot(c_fir(1:sz_a_x,:),c_sec(1:sz_a_x,:),'r*',c_fir(sz_a_x+1:sz_a_x+sz_b_x,:),c_sec(sz_a_x+1:sz_a_x+sz_b_x,:),'b*')

%% 2d-Gaussian-Distribution
% data A
mu_a = mean(pca_a);
Sigma_a = cov(pca_a);
x1 = pca_a(1:sz_a_x,1)';
x1 = sort(x1);
x2 = pca_a(1:sz_a_x,2)';
x2 = sort(x2);
[X1,X2] = meshgrid(x1,x2);
G_a = mvnpdf([X1(:) X2(:)],mu_a,Sigma_a);
G_a = reshape(G_a,length(x2),length(x1));
figure(2);
mesh(x1,x2,G_a);

% data B
mu_b = mean(pca_b);
Sigma_b = cov(pca_b);
x1 = pca_b(1:sz_b_x,1)';
x1 = sort(x1);
x2 = pca_b(1:sz_b_x,2)';
x2 = sort(x2);
[X1,X2] = meshgrid(x1,x2);
G_b = mvnpdf([X1(:) X2(:)],mu_b,Sigma_b);
G_b = reshape(G_b,length(x2),length(x1));
figure(3);
mesh(x1,x2,G_b);

%% data A, dataB, test plot
c_test1 = test*v(:,4);
c_test2 = test*v(:,3);
c_test = [c_test1, c_test2];
figure(4);
plot(c_fir(1:sz_a_x,:),c_sec(1:sz_a_x,:),'r*', c_fir(sz_a_x+1:sz_a_x+sz_b_x,:),c_sec(sz_a_x+1:sz_a_x+sz_b_x,:),'b*', c_test1, c_test2, 'g*')

%% Mahalanobis Distance
M_D_1_a = (c_test(1,:)-mu_a)*inv(Sigma_a)*(c_test(1,:)-mu_a).';
M_D_2_a = (c_test(2,:)-mu_a)*inv(Sigma_a)*(c_test(2,:)-mu_a).';
M_D_1_b = (c_test(1,:)-mu_b)*inv(Sigma_b)*(c_test(1,:)-mu_b).';
M_D_2_b = (c_test(2,:)-mu_b)*inv(Sigma_b)*(c_test(2,:)-mu_b).';
 


