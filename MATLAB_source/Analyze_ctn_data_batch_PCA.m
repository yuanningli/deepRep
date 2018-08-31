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
        mu = mean(temp.data,2);
        sd = std(temp.data,0,2);
        tdata = normcdf(temp.data,repmat(mu,1,size(temp.data,2)),repmat(sd,1,size(temp.data,2)));
        
        data_train = cat(2,data_train,tdata(:,1:2:end));
        label_train = cat(2,label_train,temp.label(:,1:2:end));
    end
end

fs = 500;
trial_time_length = 0.200;  % 200 ms
trial_sample_length = round(fs*trial_time_length);  % # of time points per trial
slid_time = 0.02;  % 20 ms
factor = trial_time_length/slid_time;
N = size(data_train,2);
num_trials = floor(N/(trial_sample_length/factor))-factor+1; % 200 ms trials with 50ms step size
batch_size = 1024;
num_batches = floor(num_trials/batch_size);

channels = 1:32;
slid = trial_sample_length/factor;
mean_vec = zeros(1,length(channels)*trial_sample_length);
cov_vec = zeros(length(channels)*trial_sample_length);

% compute mean
trial_cnt = 0;
ptr = 0;
while trial_cnt < num_trials
    current_trial = data_train(channels,(ptr+1) : (ptr + trial_sample_length))';
    current_trial = reshape(current_trial,1,[]);
    mean_vec = mean_vec + current_trial;
    trial_cnt = trial_cnt + 1;
    ptr = ptr + slid;
end 
mean_vec = mean_vec / trial_cnt;

% compute var
trial_cnt = 0;
ptr = 0;
while trial_cnt < num_trials
    current_trial = data_train(channels,(ptr+1) : (ptr + trial_sample_length))';
    current_trial = reshape(current_trial,1,[]);
    cov_vec = cov_vec + (current_trial - mean_vec)' * (current_trial - mean_vec);
    trial_cnt = trial_cnt + 1;
    ptr = ptr + slid;
end 
cov_vec = cov_vec / trial_cnt;

[V, D] = eigs(cov_vec, 20);
csvwrite(['Train_data_normed_batch_PC20.dat'],V)
csvwrite(['Train_data_normed_batch_mean.dat'],mean_vec')
csvwrite(['Train_data_normed_batch_cov.dat'],cov_vec)


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