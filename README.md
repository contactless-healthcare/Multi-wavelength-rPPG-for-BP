For dataset:
To protect the privacy of the subjects, we only uploaded raw rPPG waveforms instead of images. 
Our data contains a multichannel rPPG signal from a subject, blood pressure labels, and a timestamp corresponding to the generated label.
Moreover, the label labels of the existing data are not continuous. 
We are about to upload continuous blood pressure data from the ICU portion for other types of studies.


For code:
Since we employ LOSOCV, it is important that the id labels given to each subject are not repeated to avoid data leakage.
signal_preprocessing: You need to run "extract_sample_ICU" or "extract_sample_LAB" first, then you need to run "prepare_data" to generate the separate channel or the fusion channel. H5 file for neural network training.
training_and_predicting: "train_rppg_cycle_separate.py"  is used to train the network for the individual channels, and "train_rppg_channel_fusion.py" is used to train the wang'lu oh for the fused channels.

