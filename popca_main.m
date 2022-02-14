data=readtable('experiment_data.csv');
data=data{:,:};
X=data(:,1:size(data,2)-1);
Y=data(:,size(data,2));

N = size(X,1);
d = size(X,2);
Y_hat=X*pinv(X)*Y;
AyX=X-Y_hat*pinv(Y_hat)*X;

S_o=1/N*AyX'*AyX;
[V, D]=eig((S_o+S_o')/2);
[alleigs, order]=sort(abs(diag(D)),'descend');

qq=100;
map_q=zeros(qq,1);
pen=zeros(qq,1);
llh=zeros(qq,1);
for i=1:qq
    q=i;
    pen(i,1)=-log(N)*(q+d*q-q*(q+1)/2)/2;
    llh(i,1)=-N/2*sum(log(alleigs(1:q)))-N*(d-q)/2*log(mean(alleigs(q+1:d)));
    map_q(i,1)=llh(i,1)+pen(i,1);
end
plot(map_q)
plot(log(alleigs(1:qq)))

q=20;
wo_hat = V(:,1:q) * (diag(alleigs(1:q)) - mean(alleigs(q+1:d)) * eye(q));
wz=wo_hat*wo_hat'*X';
%wz=X'-wo_hat*wo_hat'*X';
S_p=1/N*(X'-wz)*(X'-wz)';
%S_p=1/N*wz*wz';
[Vp, Dp] = eig((S_p+S_p')/2);
%alleigsp = abs(diag(Dp));
[alleigsp, order]=sort(abs(diag(Dp)),'descend');

rr=100;
map_r=zeros(rr,1);
pen=zeros(rr,1);
llh=zeros(rr,1);
for i=1:rr
    r=i;
    pen(i,1)=-log(N)*(d*r-r*(r+1)/2)/2;
    llh(i,1)=-N/2*sum(log(alleigsp(1:r)))-N*(d-r)/2*log(mean(alleigsp(r+1:d)));
    map_r(i,1)=llh(i,1)+pen(i,1);
end
plot(map_r)
plot(log(alleigsp(1:rr)))

r=11;
wp_hat = Vp(:,1:r) * (diag(alleigsp(1:r)) - mean(alleigsp(r+1:d)) * eye(r));
xp_hat = (wp_hat'*(eye(d)-wo_hat*wo_hat')*X')';