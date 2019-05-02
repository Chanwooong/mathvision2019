clear all;
filename = 'hw10_sample.png';
img=imread(filename);
[h,w] = size(img);
input=reshape(img', h*w,1);
input= double(input);
est = zeros(h*w,6);
for n = 1:h*w
    x=rem(n,w);
    if (x==0)
        x= w;
    end
    y=fix(n/(w+1))+1; 
    est(n,1) = x*x;
    est(n,2) = y*y;
    est(n,3) = x*y;
    est(n,4) = x;
    est(n,5) = y;
    est(n,6) = 1;
end
tmp = pinv(est);
p = pinv(est)*input;
x=0;
y=0;

est_2 =zeros(h,w);
for n = 1:h*w
   x=rem(n,w);
    if (x==0)
        x= w;
    end
    y=fix(n/(w+1))+1;
    est_2(y,x) = p(1,1)*x*x + p(2,1)*y*y+p(3,1)*x*y + p(4,1)*x+p(5,1)*y+p(6,1);
end
est_2 = uint8(est_2);
imwrite(est_2, 'est.png');

sub = est_2-img;
imwrite(sub, 'sub.png');


