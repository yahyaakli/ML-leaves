function features=extractFeatures(img)

% 
% figure(1), imshow(img);
img=double(img)/255;


%% Binarization
level = graythresh(img);
imgBW = imbinarize(img,level);
imgBW=1-imgBW(:,:,1);
% figure(2), imshow(imgBW);

BW3 = bwmorph(imgBW(:,:,1),'remove');
% figure
% imshow(BW3)

%% erosion-dilation
se = strel('disk',4,4);
imgBWd=imerode(imgBW,se);
% figure,
% imshow(imgBWd);
se = strel('disk',8,8);
imgBWe=imdilate(imgBWd,se);
% figure,
% imshow(imgBWe);
se = strel('disk',4,4);
imgBWe=imerode(imgBWe,se);
% figure,
% imshow(imgBWe);

imgCropped=double(img).*repmat(double(imgBWe),1,1,3);
% figure,
% imshow(imgCropped);

%% elliptical Box

%extreme left point
xmin=find(imgBWe,1,'first');
xxmin=ceil(xmin/size(imgBWe,1));
yxmin=rem(xmin-1,size(imgBWe,1))+1;

%extreme right point
xmax=find(imgBWe,1,'last');
xxmax=ceil(xmax/size(imgBWe,1));
yxmax=rem(xmax-1,size(imgBWe,1))+1;

%highest point
ymin=find(imgBWe',1,'first');
yymin=ceil(ymin/size(imgBWe,2));
xymin=rem(ymin-1,size(imgBWe,2))+1;

%lowest point
ymax=find(imgBWe',1,'last');
yymax=ceil(ymax/size(imgBWe,2));
xymax=rem(ymax-1,size(imgBWe,2))+1;

width=xxmax-xxmin;
height=yymax-yymin;

center=[xxmin+width/2,yymin+height/2];

% Iw=find(1-imgBWe);
% center_mass(1)=mean(ceil(Iw/size(imgBWe,1)));
% center_mass(2)=mean(rem(Iw-1,size(imgBWe,1)));

%% rotation if necessary (using PCA)

[row,cols]=find(imgBWe);
iWP=[row,cols];
iWP=iWP-mean(iWP);
coviWP=iWP'*iWP;
[U,D,V]=eig(coviWP);

% if width>height
if abs(V(1,1))>abs(V(1,2))
    imgBWe=imgBWe';
    BW3=BW3';
    xyperm=[yymax yymin xxmax xxmin];
    xxmax=xyperm(1); xxmin=xyperm(2); yymax=xyperm(3); yymin=xyperm(4);
    whperm=[width height];
    height=whperm(1); width=whperm(2);
    center=[xxmin+width/2,yymin+height/2];
end

% figure, imshow(imgBWe);
% hold,
t=-pi:0.01:pi;
x_el=center(1)+width/2*cos(t);
y_el=center(2)+height/2*sin(t);
plot(x_el,y_el)




%% Morphological metric

 Iw=find(imgBWe);

% aera of leaf/area of ellipse
surface_ratio=length(Iw)/(pi*width*height)*4;

% aspect_ratio
aspect_ratio=width/height;



% leaf perimeter/leaf area
lptla_ratio=sum(sum(BW3(2:end-1,2:end-1)))/length(Iw);

% leaf perimeter/ellipse perimeter (~=pi*sqrt(2*(a^2+b^2)) according to Ramanujan approximation)
lptep_ratio=sum(sum((BW3(2:end-1,2:end-1))))/(pi*sqrt((height^2+width^2)/2));

% distance Map x
xline=linspace(xxmin,xxmax,11);
xdistmap=sum(imgBWe(2:end-1,round(xline(2:end-1))),1)/height;

% distance Map y
yline=linspace(yymin,yymax,11);
ydistmap=sum(imgBWe(round(yline(2:end-1)),2:end-1),2)/width;

% centroid radial distance
angle=-pi/2:pi/8:pi/2-pi/8;
x_coord=(1:size(imgBWe,2));
y_coord=(x_coord-center(1)).*(sin(angle')./cos(angle'))+center(2);
y_coord(1,:)=linspace(1,size(imgBWe,1),size(imgBWe,2)); % cas pi/2 ? part;
x_coord(2:length(angle),:)=repmat(x_coord,length(angle)-1,1);
x_coord(1,:)=center(1);
% figure,
% imshow(imgBWe);
% hold on
for i=1:length(angle)
    plot(x_coord(i,:),y_coord(i,:));
end


% dist to leaf edge
Iw3=find(BW3(2:end-1,2:end-1));
xedge=ceil(Iw3/size(BW3,1)-1)+2;
yedge=rem(Iw3-1,size(BW3,1)-2)+2;
scatter(xedge,yedge,'.')
coord_intersect=zeros(2*length(angle),2);
for i=1:length(angle)
    dist=(xedge-x_coord(i,:)).^2+(yedge-y_coord(i,:)).^2;
    dist=reshape(dist,length(xedge)*size(x_coord,2),1);
    [sortDist,I]=sort(dist);
    I12=rem((I-1),length(xedge))+1;
    I12=I12(abs(I12-I12(1))==0|abs(I12-I12(1))>min(width,height)/2|abs(yedge(I12)-yedge(I12(1)))>height/2);
    I12=unique(I12,'stable');
    I12=I12(1:2);
    coord_intersect(2*i-1:2*i,:)=[xedge(I12) yedge(I12)];
end
scatter(coord_intersect(:,1),coord_intersect(:,2),'o','filled')

distcentroidtole=sqrt(sum((coord_intersect-center).^2,2));

% dist to square boundary box
% plot(x_el,y_el,'LineWidth',2);
coord_intersect_bb=zeros(2*length(angle),2);
coord_intersect_bb(1:2,:)=[center(1) yymin;center(1) yymax];
for i=2:length(angle)
    coeff=sin(angle(i))./cos(angle(i));
    if coeff>0
        yxmin_int=(xxmin-center(1))*coeff+center(2);
        if yxmin_int<yymin
            xymin_int=(yymin-center(2))/coeff+center(1);
            xymax_int=(yymax-center(2))/coeff+center(1);
            coord_intersect_bb(2*i-1:2*i,:)=[xymin_int,yymin;xymax_int,yymax];
        else            
            yxmax_int=(xxmax-center(1))*coeff+center(2);
            coord_intersect_bb(2*i-1:2*i,:)=[xxmin,yxmin_int;xxmax,yxmax_int];
        end
    else
        yxmax_int=(xxmin-center(1))*coeff+center(2);
        if yxmax_int>yymax
            xymin_int=(yymax-center(2))/coeff+center(1);
            xymax_int=(yymin-center(2))/coeff+center(1);
            coord_intersect_bb(2*i-1:2*i,:)=[xymin_int,yymax;xymax_int,yymin];
        else            
            yxmin_int=(xxmax-center(1))*coeff+center(2);
            coord_intersect_bb(2*i-1:2*i,:)=[xxmin,yxmax_int;xxmax,yxmin_int];
        end
    end
end
scatter(coord_intersect_bb(:,1),coord_intersect_bb(:,2),'o','filled')
distcentroidtobb=sqrt(sum((coord_intersect_bb-center).^2,2));

% ratio
ratio_disttocentroid=distcentroidtole./distcentroidtobb;

%% Color metric

%%% TO DO


%% Vector of features

features=[surface_ratio aspect_ratio lptla_ratio lptep_ratio xdistmap ydistmap' ratio_disttocentroid'];

close all;