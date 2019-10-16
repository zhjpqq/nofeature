%% ��ȡ�����ļ�
TransformTypes = { 'Rotate', 'Zoom', 'View', 'Blur', 'Noise', 'Light' };
TransformTypesAxis = { 'Rotate Angle', 'Zoom Scale', 'Viewing Angle', 'Blur Scale', 'Noise Strengten', 'Brightness Change' };
DATASET_NAMES = { 'bike','boat', 'graf', 'bark', 'tree', 'ubc', 'wall', 'leuven' };
TransScaleSet = { RotateAngleSet, ZoomScaleSet,ViewAngleSet ,GaussianScaleSet ,NoiseCountSet ,LightnessChangeSet}; 

dir = 'F:\\MatchingPrecisionProject - MGHF\\MatchingPrecisionProject\\evaluate_results_800_cross_best\\';

% matchType = 'BruteForce';
matchType = 'CrossCheck';
% matchType = 'DistThreshold';
% 
%%%-----��һ������-------------
featureNames ={'MGHF','Feature2D.ORB','Feature2D.BRISK','Feature2D.SIFT','Feature2D.SURF'};
RotateAngleSet = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 120 ];
ZoomScaleSet =   [ 0.6 , 0.9, 1.2, 1.5,1.8 ];
ViewAngleSet =   [ 10, 20, 30, 40, 50 ];
GaussianScaleSet = [ 1.0, 3.0, 5.0, 7.0, 9.0 ];
NoiseCountSet = [ 0.01, 0.03, 0.05, 0.07, 0.09 ];
LightnessChangeSet = [ -25, -10, 5, 20, 35 ];

%%------�ڶ�������---------------
% featureNames ={'Feature2D.NOF','Feature2D.ORB','Feature2D.BRISK','Feature2D.SIFT','Feature2D.SURF'};
% RotateAngleSet = [ 20, 40, 60, 80, 100, 120, 140, 160,180 ];
% ZoomScaleSet =   [  0.4,0.8,1.2,1.6,2.0,2.4 ];
% ViewAngleSet =   [ 5, 15, 25, 30, 35 ];
% GaussianScaleSet = [ 1.0, 3.0, 5.0, 7.0, 9.0,11.0 ];
% NoiseCountSet = [ 0.01, 0.03, 0.05, 0.07, 0.09 ];
% LightnessChangeSet = [ -25, -10, 5, 20, 35 ];


Marks = { 'r-*' , 'b-^' ,'g-o','c->','k-+'};

%%��ͼ
for k=1:length(TransformTypes)
    for j=1:length(featureNames)
        filename = [dir matchType '_' TransformTypes{k} '_'  featureNames{j} '.csv'] ;
        data=load(filename)/10000;
        for i=1:length(DATASET_NAMES)
            figure(k);set(gcf,'color','w'); hold on;
            subplot(2,4,i);
            plot(TransScaleSet{k},data(i,:),Marks{j});
            axis([TransScaleSet{k}(1), TransScaleSet{k}(end),0,1]);        
            title(DATASET_NAMES(i)); xlabel(TransformTypesAxis{k});ylabel('Precision');
        end;
         legend(featureNames);
    end

end


% for j=1:length(featureNames)
%     filename = [dir matchType '_' transType '_'  featureNames{j} '.csv'] ;
%     data=load(filename)/10000;
%     for i=1:length(DATASET_NAMES)
%         figure(1); hold on;
%         subplot(2,4,i);
%         plot(RotateAngleSet,data(i,:),Marks{j});
%         axis([RotateAngleSet(1), RotateAngleSet(end),0,1]);        
%         title(DATASET_NAMES(i)); xlabel('Rotate Angle');ylabel('Precision');
%     end;
%     legend(featureNames{j});
% end