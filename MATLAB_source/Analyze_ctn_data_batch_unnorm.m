clear all

addpath('/media/RAWDATA/ECOG/Participants/P41_0319/Preprocessed/ctnData/')
cd('/home/yuanning/python/deeprep/data/train')
% task_list = {'CL','CW','ECoG_bodies','EmoLoc','FBL','FWL','KDEFg','PL','PosImp','RaFD','RedGreen','SCV','SD1','SD2','WC'};
% num_sessions = [1,1,2,1,1,1,2,2,1,3,1,2,1,1,2];
task_list = {'CL','CW','ECoG_bodies','EmoLoc','KDEFg','PL','PosImp','RaFD','RedGreen','SCV','SD','WC'};
num_sessions = [1,1,2,1,2,2,1,3,1,2,2,2];

data_train = [];
label_train = [];
for i = 1 : length(task_list)
    for j = 1 : num_sessions(i)
        temp = load(['P41_',task_list{i},num2str(j),'_raw_LFP_1000.mat']);
        fprintf(['P41_',task_list{i},num2str(j),'_raw_LFP_1000.mat \n'])
        temp.label = zeros(1,size(temp.data,2));
        for n = 1 : length(temp.Time_onsets)
            temp.label(temp.Time_onsets(n) : (temp.Time_onsets(n) + 500)) = 1;
        end
        
        data_train = cat(2,data_train,temp.data(:,1:2:end));
        label_train = cat(2,label_train,temp.label(:,1:2:end));
    end
end
%%
channel = [9,10,11,12,33,34,35,36,71,72,73,74,91,92,93,94];
for i = 1 : 16
    subplot(4,4,i)
    histogram(data_train(channel(i),:),'Normalization','pdf')
    hold on
    xlim([-600,600])
    m = mean(data_train(channel(i),:));
    s = std(data_train(channel(i),:));
    plot(-600:600, normpdf(-600:600, m, s),'linewidth',2)
end
