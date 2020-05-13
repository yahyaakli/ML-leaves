function [erreur,errClass] = calculerErreur(Vmb)
    errClass = [];
    Label = [ones(1,5),2*ones(1,5),3*ones(1,5),4*ones(1,5),5*ones(1,5)];
    erreur = 0;
    for i=1:size(Vmb,2)
       S = 0;
       for j=1:size(Vmb,1)
           if(j~=Label(i))
                S=S+Vmb(j,i);
           end 
       end
       S=S/4;
       errClass = [errClass,S];
       erreur = erreur + S;
    end
end