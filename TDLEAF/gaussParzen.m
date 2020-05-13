function d =  gaussParzen(x,Xk,sigma)
    sig = (sigma.^2)*eye(size(Xk,2));
    d=0;
    for j=1:size(Xk,1)
        d=d+mvnpdf(x',Xk(j,:),sig);
    end
    d=d/size(Xk,1);
end