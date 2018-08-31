clear all

addpath('/media/RAWDATA/ECOG/Participants/P41_0319/Preprocessed/ctnData/')
cd('/home/yuanning/python/deeprep/data/test')
% task_list = {'CL','CW','ECoG_bodies','EmoLoc','FBL','FWL','KDEFg','PL','PosImp','RaFD','RedGreen','SCV','SD1','SD2','WC'};
% num_sessions = [1,1,2,1,1,1,2,2,1,3,1,2,1,1,2];
% task_list = {'CL','CW','ECoG_bodies','EmoLoc','KDEFg','PL','PosImp','RaFD','RedGreen','SCV','SD','WC'};
% num_sessions = [1,1,2,1,2,2,1,3,1,2,2,2];
task_list = {'FWL'};
num_sessions = [1];

data_test = [];
label_test = [];
for i = 1 : length(task_list)
    if strcmp(task_list{i},'FBL')
        keys = [1,2,3,4,5,6];
    else
        keys = [1,2,7,8,3,6,4];
    end
    for j = 1 : num_sessions(i)
        temp = load(['P41_',task_list{i},num2str(j),'_raw_LFP_1000.mat']);
        fprintf(['P41_',task_list{i},num2str(j),'_raw_LFP_1000.mat \n'])
        temp.label = zeros(1,size(temp.data,2));
        for n = 1 : length(temp.Time_onsets)
            temp.label(temp.Time_onsets(n) : (temp.Time_onsets(n) + 500)) = keys(temp.Info.cond_type(n));
        end

        mu = mean(temp.data,2);
        sd = std(temp.data,0,2);
        tdata = normcdf(temp.data,repmat(mu,1,size(temp.data,2)),repmat(sd,1,size(temp.data,2)));
        data_test = cat(2,data_test,tdata(:,1:2:end));
        label_test = cat(2,label_test,temp.label(:,1:2:end));
    end
end

fs = 500;
trial_time_length = 0.200;  % 200 ms
trial_sample_length = round(fs*trial_time_length);  % # of time points per trial
slid_time = 0.02;  % 20 ms
factor = trial_time_length/slid_time;
N = size(data_test,2);
num_trials = floor(N/(trial_sample_length/factor))-factor+1; % 200 ms trials with 50ms step size
batch_size = 1024;
num_batches = floor(num_trials/batch_size);

batch_cnt = 0;
ptr = 0;
slid = trial_sample_length/factor;
while batch_cnt < num_batches
    trial_cnt = 0;
    batch_data = [];
    batch_label = [];
    while trial_cnt < batch_size
        current_trial = data_test(:,(ptr+1) : (ptr + trial_sample_length))';
        current_label = label_test((ptr+1) : (ptr + trial_sample_length))';
        if length(find(current_label)) >= length(current_label)/2
            ulabel = unique(current_label);
            ulabel = ulabel(ulabel ~= 0);
            current_index = ulabel(1);
        else
            current_index = 0;
        end
        batch_label = cat(1,batch_label,current_index);
        batch_data = cat(1,batch_data,current_trial);
        trial_cnt = trial_cnt + 1;
        ptr = ptr + slid;
    end
    csvwrite(['Test_data_normed_',task_list{1},'_batch',num2str(batch_cnt),'.dat'],batch_data)
    csvwrite(['Test_data_',task_list{1},'_label_batch',num2str(batch_cnt),'.dat'],batch_label)
    fprintf('writing batch #%d...\n',batch_cnt)
    batch_cnt = batch_cnt + 1;
end 


% csvwrite(['Train_data_500hz.dat'],data_train)
% data_test = [];
% for i = 1 : length(test_list)
%     for j = 1 : num_sessions_test(i)
%         temp = load(['P41_',task_list{i},num2str(j),'_raw_LFP_1000.mat']);
%         fprintf(['P41_',task_list{i},num2str(j),'_raw_LFP_1000.mat \n'])
%         data_test = cat(2,data_test,temp.data(:,1:2:end));
%     end
% end
% csvwrite(['Test_data_500hz.dat'],data_test)