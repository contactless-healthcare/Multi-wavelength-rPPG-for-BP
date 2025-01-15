% This code is used to consolidate the data into a single.h5 file
% that was included during the training


clear all;
close all;


rootDir = 'E:\XX\XX\XX\';
NumberDir = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',...
'21','22','23','24','25','26','27','28','29','30'};

filePath_1 =  'XX\XX\';
g_filename = 'cycle_g.xlsx';
r_filename = 'cycle_r.xlsx';
ir_filename = 'cycle_ir.xlsx';


h5filename = 'XX.h5';


allData_g = [];
allData_r = [];
allData_ir = [];
Datacount = [];

for i = 1:length(NumberDir)
    
    data_g = xlsread(fullfile(rootDir,NumberDir{i},filePath_1,g_filename));
    data_r = xlsread(fullfile(rootDir,NumberDir{i},filePath_1,r_filename));
    data_ir = xlsread(fullfile(rootDir,NumberDir{i},filePath_1,ir_filename));
    count = size(data_g, 1);
    
    allData_g = [allData_g; data_g];
    allData_r = [allData_r; data_r];
    allData_ir = [allData_ir; data_ir];
    Datacount = [Datacount;count];
end

subject_id = allData_g(:, end);
subject_id = transpose(subject_id);

label_colums = size(allData_g, 2) - 3 : size(allData_g, 2) - 1;
groundtruth = allData_g(:, label_colums);
groundtruth = transpose(groundtruth);

num_cols = size(allData_g, 2);

rppg_g = allData_g(:, 1:(num_cols - 4));
rppg_r = allData_r(:, 1:(num_cols - 4));
rppg_ir = allData_ir(:, 1:(num_cols - 4));

%% for single channel
rppg_g = transpose(rppg_g);
rppg_r = transpose(rppg_r);
rppg_ir = transpose(rppg_ir);
h5create(h5filename,'/ppg',size(rppg_g));
h5write(h5filename,'/ppg',rppg_g);

%% for multi channel
num = size(rppg_g, 2);
ppg = zeros(50, num, 2);
ppg(:,:,1) = rppg_g;
ppg(:,:,2) = rppg_r;
ppg(:,:,3) = rppg_ir;
h5create(h5filename,'/ppg',size(ppg));
h5write(h5filename,'/ppg',ppg);

%% for id and gt
h5create(h5filename,'/label',size(groundtruth));
h5write(h5filename,'/label',groundtruth);
h5create(h5filename,'/subject_idx',size(subject_id));
h5write(h5filename,'/subject_idx',subject_id);

