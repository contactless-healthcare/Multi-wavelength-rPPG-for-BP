clear all;
close all;
clc;

Rootpath = 'E:\XX\XX\XX\';
NumberDir = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
save_path_1 = 'E:\XX\XX\XX\';
save_path_2 = 'XX';


FS = 30;
window_length = 50;
overlap_percentage = 20;
overlap_length = 10;

for k = 1:length(NumberDir)
    subject_path = NumberDir{k};
    dir_label = fullfile(Rootpath,NumberDir{k},'血压记录.xlsx');
    currentDir = fullfile(Rootpath, NumberDir{k});  
    signaldirR = fullfile(currentDir, "R.mat");  
    signaldirG = fullfile(currentDir, "G.mat");  
    signaldirIR = fullfile(currentDir, "IR.mat");  
    

    TestData = load(signaldirR);
    R_signal_all = TestData.pixelR1;
    TestData = load(signaldirG);
    G_signal_all = TestData.pixelG1; 
    TestData = load(signaldirIR);
    IR_signal_all = TestData.pixelIR1;
    
    
    Labels = {'time', 'SBP', 'DBP'};
    dataTable = readtable(char(dir_label), 'ReadRowNames', true);
    timepoints = dataTable{Labels{1}, :};
    SBP_data = dataTable{Labels{2}, :};
    DBP_data = dataTable{Labels{3}, :};
    MBP_data = (SBP_data + (2 * DBP_data)) / 3;
    

    total_sample_r=[];
    total_sample_g=[];
    total_sample_ir=[];
    
    %DC norm
    R_signal_all = R_signal_all ./ mean(R_signal_all) - 1.0;
    G_signal_all = G_signal_all ./ mean(G_signal_all) - 1.0;
    IR_signal_all = IR_signal_all ./ mean(IR_signal_all) - 1.0;
    
    %Band-pass
    bandwidth1=[30/60, 360/60];
    [n, d] = butter(4, bandwidth1/(FS/2), 'bandpass');
    R_signal_all = filtfilt(n,d,R_signal_all')';
    G_signal_all = filtfilt(n,d,G_signal_all')'; 
    IR_signal_all = filtfilt(n,d,IR_signal_all')';
       



    signal_length = length(G_signal_all);

    %calculate SNR of a signal part, and pass the part with SNR below -7
    SNR_ir = [];
    SNR_r = [];
    SNR_g = [];
    segmentLength = 500;
    
    for i = 1:length(timepoints)
        if timepoints(i)+500 <= signal_length
            segment = G_signal_all(timepoints(i)-500:timepoints(i)+500);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_g = [SNR_g, SNR];
        else
            segment = G_signal_all(timepoints(i)-500:signal_length);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_g = [SNR_g, SNR];
        end
    end

     for i = 1:length(timepoints)
        if timepoints(i)+500 <= signal_length
            segment = IR_signal_all(timepoints(i)-500:timepoints(i)+500);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_ir = [SNR_ir, SNR];
        else
            segment = IR_signal_all(timepoints(i)-500:signal_length);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_ir = [SNR_ir, SNR];
        end
     end

     for i = 1:length(timepoints)
        if timepoints(i)+500 <= signal_length
            segment = R_signal_all(timepoints(i)-500:timepoints(i)+500);   
            HR = prpsd(segment, FS, 30, 360, 0);
            SNR = bvpsnr(segment, FS, HR, 0);
            SNR_r = [SNR_r, SNR];
        else
            segment = R_signal_all(timepoints(i)-500:signal_length);   
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
        start_index = timepoints(m)-500;
        end_index = timepoints(m)+500;
        
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
        
        %segment
        for i = 1:overlap_length:length(G_signal_part)-window_length+1
                
                start_index = i;
                end_index = i + window_length - 1;
                current_segment_G = G_signal_part(start_index:end_index); 
                min_value_G = min(current_segment_G(:));
                max_value_G = max(current_segment_G(:));
                current_segment_G = 2 * (current_segment_G - min_value_G) / (max_value_G - min_value_G) - 1;
                      
                current_segment_R = R_signal_part(start_index:end_index);
                min_value_R = min(current_segment_R(:));
                max_value_R = max(current_segment_R(:));
                current_segment_R = 2 * (current_segment_R - min_value_R) / (max_value_R - min_value_R) - 1;
                
                current_segment_IR = IR_signal_part(start_index:end_index);
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
        
        len_signal = window_length + 4;
        data_IR = zeros(size(cycle_IR,1), len_signal);
        data_R  = zeros(size(cycle_IR,1), len_signal);
        data_G  = zeros(size(cycle_IR,1), len_signal);
        
        for i = 1:size(cycle_IR,1)
            data_IR(i, 1:window_length) = cycle_IR(i, :);
            data_IR(i, window_length + 1:end) = [SBP_data(m), DBP_data(m), MBP_data(m), str2double(NumberDir{k})];
        end
        
        for i = 1:size(cycle_R,1)
            data_R(i, 1:window_length) = cycle_R(i, :);
            data_R(i, window_length + 1:end) = [SBP_data(m), DBP_data(m), MBP_data(m), str2double(NumberDir{k})];
        end
        
        for i = 1:size(cycle_G,1)
            data_G(i, 1:window_length) = cycle_G(i, :);
            data_G(i, window_length + 1:end) = [SBP_data(m), DBP_data(m), MBP_data(m), str2double(NumberDir{k})];
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
