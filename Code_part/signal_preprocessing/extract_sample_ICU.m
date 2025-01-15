clear all;
close all;
clc;

Rootpath = 'E:\XX\XX\XX\';
NumberDir = {'21','22','23','24','25','26','27','28','29','30'};
save_path_1 = 'E:\XX\XX\XX\';
save_path_2 = 'XX';

for k = 1:length(NumberDir)
    subject_path = NumberDir{k};
    data_label = readmatrix(fullfile(Rootpath, subject_path,'label_frame.csv'), 'NumHeaderLines', 1);
    dir = fullfile(Rootpath, subject_path);
    signaldirR =  fullfile(Rootpath, subject_path,"R.mat");
    signaldirG = fullfile(Rootpath, subject_path,"G.mat");
    signaldirIR = fullfile(Rootpath, subject_path,"IR.mat");
    TestData = load(signaldirR);
    rppg_r = TestData.R;
    TestData = load(signaldirG);
    rppg_g = TestData.G;
    TestData = load(signaldirIR);
    rppg_ir = TestData.IR;
    
    FS = 60;
    window_length = 100;
    resize_length = 50;
    overlap_percentage = 20;
    overlap_length = 20;
    
    total_sample_r=[];
    total_sample_g=[];
    total_sample_ir=[];
    
    rppg_r = rppg_r - rppg_ir;
    rppg_g = rppg_g - rppg_ir;
    
    rppg_r = rppg_r ./ mean(rppg_r) - 1.0;
    rppg_g = rppg_g ./ mean(rppg_g) - 1.0;
    rppg_ir = rppg_ir ./ mean(rppg_ir) - 1.0;
    
    bandwidth1=[30/60, 360/60];
    [n, d] = butter(4, bandwidth1/(FS/2), 'bandpass');
    R_signal_all = filtfilt(n,d,rppg_r')';
    G_signal_all = filtfilt(n,d,rppg_g')'; 
    IR_signal_all = filtfilt(n,d,rppg_ir')';
    
    timepoints = data_label(:, 1);
    SBP_data = data_label(:, 2);
    DBP_data = data_label(:, 3);
    MBP_data = data_label(:, 4);
    signal_length = length(rppg_g);
    
    SNR_ir = [];
    SNR_r = [];
    SNR_g = [];
    segmentLength = 1800;
    
    for i = 1:segmentLength:signal_length
        if i + segmentLength - 1 <= signal_length
            segment = G_signal_all(i+50:i+segmentLength-50);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_g = [SNR_g, SNR];
        end
    end

    for i = 1:segmentLength:signal_length
        if i + segmentLength - 1 <= signal_length
            segment = IR_signal_all(i+50:i+segmentLength-50);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_ir = [SNR_ir, SNR];
        end
    end

    for i = 1:segmentLength:signal_length
        if i + segmentLength - 1 <= signal_length
            segment = R_signal_all(i+50:i+segmentLength-50);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_r = [SNR_r, SNR];
        end
    end

    abandon_g = find(SNR_g < -7);
    abandon_r = find(SNR_r < -7);
    abandon_ir = find(SNR_ir < -7);
    abandon_part = union(abandon_g, abandon_r);
    abandon_part = union(abandon_part,abandon_ir);
    
    for m = 1:length(timepoints)
        if ismember(m, abandon_part)
            continue;
        end
        start_index = (m-1)*1800+50;
        end_index = m*1800-50;
        
        if start_index > 0 && end_index <= length(G_signal_all)
            G_signal_part = G_signal_all(start_index:end_index);
            R_signal_part = R_signal_all(start_index:end_index);
            IR_signal_part = IR_signal_all(start_index:end_index);
        else
            G_signal_part = G_signal_all(start_index:length(G_signal_all));
            R_signal_part = R_signal_all(start_index:length(G_signal_all));
            IR_signal_part = IR_signal_all(start_index:length(G_signal_all));
        end
        
        cycle_G = [];
        cycle_R = [];
        cycle_IR = [];
        
        for i = 1:overlap_length:length(G_signal_part)-window_length+1
                
                start_index = i;
                end_index = i + window_length - 1;
                
                current_segment_G = G_signal_part(start_index:end_index);
                current_segment_G = interp1(1:length(current_segment_G), current_segment_G, linspace(1, length(current_segment_G), resize_length), 'pchip');
                min_value_G = min(current_segment_G(:));
                max_value_G = max(current_segment_G(:));
                current_segment_G = 2 * (current_segment_G - min_value_G) / (max_value_G - min_value_G) - 1;
                      
                current_segment_R = R_signal_part(start_index:end_index);
                current_segment_R = interp1(1:length(current_segment_R), current_segment_R, linspace(1, length(current_segment_R), resize_length), 'pchip');
                min_value_R = min(current_segment_R(:));
                max_value_R = max(current_segment_R(:));
                current_segment_R = 2 * (current_segment_R - min_value_R) / (max_value_R - min_value_R) - 1;
                
                current_segment_IR = IR_signal_part(start_index:end_index);
                current_segment_IR = interp1(1:length(current_segment_IR), current_segment_IR, linspace(1, length(current_segment_IR), resize_length), 'pchip');
                min_value_IR = min(current_segment_IR(:));
                max_value_IR = max(current_segment_IR(:)); 
                current_segment_IR = 2 * (current_segment_IR - min_value_IR) / (max_value_IR - min_value_IR) - 1;
                
                
                cycle_G = [cycle_G; current_segment_G]; 
                cycle_R = [cycle_R; current_segment_R];
                cycle_IR = [cycle_IR; current_segment_IR];
                
                               
        end
        %%%——————————————————————————————————————————————
%          Here, we append the ID of each subject (i.e., the folder number of each subject's data)
%          along with the label at the end of each data.
%          In order to avoid possible data leakage during training (multiple subjects share the same ID),
%          please make sure that each subject participating in the training has a unique number
%          len_signal = resize_length + 4;
        %%%——————————————————————————————————————————————
         data_IR = zeros(size(cycle_IR,1), len_signal);
         data_R  = zeros(size(cycle_IR,1), len_signal);
         data_G  = zeros(size(cycle_IR,1), len_signal);

         for i = 1:size(cycle_IR,1)
                data_IR(i, 1:resize_length) = cycle_IR(i, :);
                data_IR(i, resize_length + 1:end) = [SBP_data(m), DBP_data(m), MBP_data(m), str2double(NumberDir{k})];
         end

         for i = 1:size(cycle_R,1)
                data_R(i, 1:resize_length) = cycle_R(i, :);
                data_R(i, resize_length + 1:end) = [SBP_data(m), DBP_data(m), MBP_data(m), str2double(NumberDir{k})];
         end

         for i = 1:size(cycle_G,1)
                data_G(i, 1:resize_length) = cycle_G(i, :);
                data_G(i, resize_length + 1:end) = [SBP_data(m), DBP_data(m), MBP_data(m), str2double(NumberDir{k})];
         end
        
         total_sample_r = [total_sample_r; data_R];
         total_sample_g = [total_sample_g; data_G];
         total_sample_ir = [total_sample_ir; data_IR];
     
    end

    xlswrite(fullfile(save_path_1,subject_path,save_path_2,'cycle_g.xlsx'),total_sample_g);
    xlswrite(fullfile(save_path_1,subject_path,save_path_2,'cycle_r.xlsx'),total_sample_r);
    xlswrite(fullfile(save_path_1,subject_path,save_path_2,'cycle_ir.xlsx'),total_sample_ir);       
    
    fprintf('finished %s\n', NumberDir{k});

    end    
    
    