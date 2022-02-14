%sampling and correct detection rate evaluation
qq=20;
rr=20;
rp=10;
nbc=zeros(rp,1);
for k=1:rp
    sample=datasample(data,60,'Replace',false);
    %sample=sample{:,:};
    X=sample(:,1:size(sample,2)-1);
    Y=sample(:,size(sample,2));
    N = size(X,1);
    d = size(X,2);
    Y_hat=X*pinv(X)*Y;
    AyX=X-Y_hat*pinv(Y_hat)*X;
    
    S_o=1/N*AyX'*AyX;
    [V, D]=eig(S_o);
    [alleigs, order]=sort(abs(diag(D)),'descend');
    
    map_q=zeros(qq,1);
    for q=1:qq
        map_q(q,1)=-N/2*sum(log(alleigs(1:q)))-N*(d-q)/2*log(mean(alleigs(q+1:d)))-log(N)*(q+d*q-q*(q+1)/2)/2;
    end
    
    C_q=max(map_q(:));
    q=find(map_q==C_q);
    wo_hat = V(:,1:q) * (diag(alleigs(1:q)) - mean(alleigs(q+1:d)) * eye(q));
    wz=wo_hat*wo_hat'*X';
    S_p=1/N*(X'-wz)*(X'-wz)';
    [Vp, Dp] = eig(S_p);
    [alleigsp, order]=sort(abs(diag(Dp)),'descend');

    map_r=zeros(rr,1);
    for r=1:rr
        map_r(r,1)=-N/2*sum(log(alleigsp(1:r)))-N*(d-r)/2*log(mean(alleigsp(r+1:d)))-log(N)*(d*r-r*(r+1)/2)/2;
    end
    
    C_r=max(map_r(:));
    r=find(map_r==C_r);
    nbc(k,1)=r;
end