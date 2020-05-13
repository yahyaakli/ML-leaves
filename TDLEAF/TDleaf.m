% clear all
close all

LeafType={'papaya','pimento','chrysanthemum','chocolate_tree'};%,...
% 'duranta_gold','eggplant','ficus','fruitcitere','geranium','guava',...
% 'hibiscus','jackfruit','ketembilla','lychee','ashanti_blood','mulberry_leaf',...
% 'barbados_cherry','beaumier_du_perou','betel','pomme_jacquot','bitter_orange',...
% 'rose','caricature_plant','star_apple','chinese_guava','sweet_olive','sweet_potato',...
% 'thevetia','coeur_demoiselle','vieux_garcon','coffee','croton'};

label=[];
X=[];
for LT=LeafType([1 2 3 4])
    
    filenames=dir([LT{1},filesep,'Training',filesep,'*.png']);
    
    
    for ifile=1:length(filenames)
        
        img=imread([filenames(ifile).folder,filesep,filenames(ifile).name]);
        X=[X;extractFeatures(img)];
        label=[label,LT];
        close all;
        
    end
end
labeltest=[];
Xtest=[];
for LT=LeafType([1 2 3 4])
    
    filenames=dir([LT{1},filesep,'Test',filesep,'*.png']);
    
    
    for ifile=1:length(filenames)
        
        img=imread([filenames(ifile).folder,filesep,filenames(ifile).name]);
        Xtest=[Xtest;extractFeatures(img)];
        labeltest=[labeltest,LT];
        close all;
        
    end
end

%% manual feature selection

Xs=X(:,1:3);
figure, hold,
for LT=LeafType
    Ilt=find(strcmp(label,LT));
    scatter3(Xs(Ilt,1),Xs(Ilt,2),Xs(Ilt,3),'o','filled');
%     pause
end
legend(LeafType(1:4),'Location','SouthWest');


%% Dimension reduction by PCA
avgX = mean(X);
X = X-avgX;
Xtest = Xtest-avgX;
covV = X'*X;
[U,D,V] = eig(covV);
[d,Isort] = sort(diag(D),'desc');
V = V(:,Isort);
Xp = X*V;
Xptest = Xtest*V;
figure, hold,
for LT = LeafType
    Ilt = find(strcmp(label,LT));
    scatter3(Xp(Ilt,1),Xp(Ilt,2),Xp(Ilt,3),'o','filled');
end
legend(LeafType(1:4),'Location','SouthWest');

%% Supervised Approaches (nonparamatric - parzen)
sig = 0.5;
% sigma = sig*eye(3);
% K=length(LeafType);
% Xtest = Xp(20,1:3);
% llh = zeros(K,1);
% for k=1:K
%    for j=1:15
%        llh(k)=llh(k)+mvnpdf(Xtest,Xp((k-1)*15+j,1:3),sigma);
%    end
%    llh(k)=llh(k)/size(Xk,1);
% end
% [mllh,c]=max(llh);
% disp(['classif result: ',LeafType(c)]);
figure; hold on;
% Training
for LT=LeafType
    Ilt=find(strcmp(label,LT));
    isoContoursParzen(Xp(Ilt,1:2),sig);
    scatter(Xp(Ilt,1),Xp(Ilt,2),'o','filled');
    
    Ilttest = find(strcmp(labeltest,LT));
    scatter(Xptest(Ilttest,1),Xptest(Ilttest,2),'o','filled');
end

% Test
K = length(LeafType);
Vsmb = zeros(K,size(Xptest,1));
for i=1:size(Xptest,1)
    for k=1:K
        Vsmb(k,i)=gaussParzen(Xptest(i,1:3)',Xp((k-1)*15+1:k*15,1:3),sig);
    end
end
[erreur,errClass] = calculerErreur(Vsmb);

%% Supervised Approaches (parametric)
% Training
d=3;
K = length(LeafType);
muT = zeros(d,K);
CovT = zeros(d,d,K);
figure; hold on;
i=1;
for LT = LeafType
    Ilt = find(strcmp(label,LT));
    muT(:,i) = mean(Xp(Ilt,1:d))';
    CovT(:,:,i) = cov(Xp(Ilt,1:d));
    dim=[1 2];   %choix de la dimension de projection
    isoContoursGauss(muT(dim,i),CovT(dim,dim,i));
    scatter(Xp(Ilt,dim(1)),Xp(Ilt,dim(2)),'o','filled');
    i=i+1;
end

K = length(LeafType);
Vsmbparametric = zeros(K,size(Xptest,1));
for i=1:size(Xptest,1)
    for k=1:K
        Vsmbparametric(k,i)=mvnpdf(Xptest(i,1:3)',muT(:,k),CovT(:,:,k));
    end
    
end

[erreurparam,errClassparam] = calculerErreur(Vsmbparametric);


%% Unsupervised Approaches: K-means and GMM-EM



%% K-means
% initialisation des centroides des classes
figure, scatter(Xp(:,1),Xp(:,2),'o','filled');
% Cinit=[-1.5 0 1;-1.5 1 -1;0 -0.4 0;1 0.6 1];
% Cinit = [0.5 -0.5 0;1 0.5 0;0 0.4 0;-1 1 0]; 
Cinit = [0.5 -0.5 0;0 0.5 0;0 0.4 0;-1 1 0]; 
% Cinit = [0.5 -0.5 0;1 0.5 0;0 0.4 0;-1 1 -1]; 
gr=[1,2,3,4];

hold;
h1=gscatter(Cinit(:,1),Cinit(:,2),gr,'rgkc','oo',20,[],'off');
set(h1,'MarkerSize',10)
h1=scatter(Cinit(1,1),Cinit(1,2),50,'r','O','filled');    
h2=scatter(Cinit(2,1),Cinit(2,2),50,'g','O','filled');
h3=scatter(Cinit(3,1),Cinit(3,2),50,'k','O','filled');
h4=scatter(Cinit(4,1),Cinit(4,2),50,'c','O','filled');
axis([-2 1.5 -0.6 1.2]);

% Mise en oeuvre des k-moyennes
% Cette boucle détail le résultat pour chaque itération
% 7 itérations sont nécessaires pour converger dans ce cas

for i=1:7
    pause(1)
    opts = statset('MaxIter',i);
    [IDX,C] = kmeans(Xp(:,1:3),4,'start',Cinit,'options',opts);
    hold off;
%     h1=gscatter(C(:,1),C(:,2),gr,'rg','OO',20,[],'off');
%     set(h1,'MarkerSize',10);
    h1=gscatter(C(:,1),C(:,2),gr,'rgkc','oo',20,[],'off');
    hold on;
    h1=scatter(C(1,1),C(1,2),50,'r','O','filled');    
    h2=scatter(C(2,1),C(2,2),50,'g','O','filled');
    h3=scatter(C(3,1),C(3,2),50,'k','O','filled');
    h4=scatter(C(4,1),C(4,2),50,'c','O','filled');
    axis([-2 1.5 -0.6 1.2]);
    gscatter(Xp(:,1),Xp(:,2),IDX,'rgkc','oo',[],'filled','off');
   
end

%% EM
clear Sigma
clear Sigmak

N=size(Xp,1);
D=3;
K=4;

% Initialisation "à la main" des moyennes et covariances des classes
mu=[-1 1 1; 0 0 1;1 1 -1;0 1 -1];
Sigma(1,:,:)=eye(D);
Sigma(2,:,:)=eye(D);
Sigma(3,:,:)=eye(D);
Sigma(4,:,:)=eye(D);
pi=ones(1,K)/K;
apost=zeros(K,N);


figure, scatter(Xp(:,1),Xp(:,2),20,'b','fill');

gr=[1,2,3,4];

hold;
h1=gscatter(mu(:,1),mu(:,2),gr,'rgbk','xx',[],'off');
set(h1,'MarkerSize',20)
h1=scatter(mu(1,1),mu(1,2),50,'r','O','filled');    
h2=scatter(mu(2,1),mu(2,2),50,'g','O','filled');
h3=scatter(mu(3,1),mu(3,2),50,'b','O','filled');
h4=scatter(mu(4,1),mu(4,2),50,'k','O','filled');

axis([-2 1.5 -0.6 1.2]);

% code de mise en oeuvre de l'EM (limité à 50 itérations)

for i=1:50
    pause(0.2)
    % �tape E
    for k=1:K
        muk=mu(k,:);
        Sigmak(:,:)=Sigma(k,:,:);
        apost(k,:)=mvnpdf(Xp(:,1:3),muk,Sigmak)*pi(k);
    end
    apost=apost./repmat(sum(apost),K,1);
    
    hold on;
%     gscatter(data(:,1),data(:,2),IDX,'rg','**',[],'off');
    
    color=[apost(1:3,:)'];
    scatter(Xp(:,1),Xp(:,2),20,color,'fill');
    axis([-2 1.5 -0.6 1.2]);
    pause(0.2)
    
    % Etape M
    for k=1:K
        mu(k,:)=sum(repmat(apost(k,:)',1,D).*Xp(:,1:3))./repmat(sum(apost(k,:)'),1,D);
        Sigma(k,:,:)=(repmat(apost(k,:)',1,D).*(Xp(:,1:3)-repmat(mu(k,:),N,1)))'*(Xp(:,1:3)-repmat(mu(k,:),N,1))./(repmat(sum(apost(k,:)'),D,D));
        pi(:,k)=sum(apost(k,:))/N;
    end
    
    % Décision du MAP pour affichage
    [~,IDX]=max(apost);
    hold off;
    h1=gscatter(mu(:,1),mu(:,2),gr,'rgbk','xx',[],'off');
    set(h1,'MarkerSize',20);
    hold on;
    h1=scatter(mu(1,1),mu(1,2),50,'r','O','filled');    
    h2=scatter(mu(2,1),mu(2,2),50,'g','O','filled');
    h3=scatter(mu(3,1),mu(3,2),50,'b','O','filled');
    h4=scatter(mu(4,1),mu(4,2),50,'k','O','filled');

%     gscatter(data(:,1),data(:,2),IDX,'rg','**',[],'off');
    color=[apost(1:3,:)'];
    scatter(Xp(:,1),Xp(:,2),20,color,'fill');
    axis([-2 1.5 -0.6 1.2]);
    hold off;
end
    
    

