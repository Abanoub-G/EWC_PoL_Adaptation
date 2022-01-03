% close all
clear all; close all; clc;

% file_name1 = 'LogFiles/results_combined_18-1.csv'; % For 01 experiments, 10 new instances with 70% on new tasks
% file_name2 = 'LogFiles/results_combined_18-2.csv'; % For 01 experiments, 10 new instances with 70% on new tasks
% file_name1 = 'LogFiles/results_combined_19-1.csv'; % For 01 experiments, 10 new instances with 95% on old tasks
% file_name2 = 'LogFiles/results_combined_19-2.csv'; % For 01 experiments, 10 new instances with 95% on old tasks
% file_name1 = 'LogFiles/results_combined_21-1.csv'; % For 01 experiments, 10 new instances with 98% on old tasks
% file_name2 = 'LogFiles/results_combined_21-2.csv'; % For 01 experiments, 10 new instances with 98% on old tasks
% file_name1 = 'LogFiles/results_combined_22-1.csv'; % For 01 experiments, 30 new instances with 95% on old tasks
% file_name2 = 'LogFiles/results_combined_22-2.csv'; % For 01 experiments, 30 new instances with 95% on old tasks
% file_name1 = 'LogFiles/results_combined_23-1.csv'; % For 01 experiments, 30 new instances with 70% on new tasks
% file_name2 = 'LogFiles/results_combined_23-2.csv'; % For 01 experiments, 30 new instances with 70% on new tasks
% file_name1 = 'LogFiles/results_combined_24-1.csv'; % For 20 experiments, 10 new instances with 95% on old tasks
% file_name2 = 'LogFiles/results_combined_24-2.csv'; % For 20 experiments, 10 new instances with 95% on old tasks
% file_name1 = 'LogFiles/results_combined_25-1.csv'; % For 20 experiments, 10 new instances with 70% on new tasks
% file_name2 = 'LogFiles/results_combined_25-2.csv'; % For 20 experiments, 10 new instances with 70% on new tasks
% file_name1 = 'LogFiles/results_combined_26-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using Grid search
% file_name2 = 'LogFiles/results_combined_26-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using Grid search
% [retraining model was not correctly assigned] file_name1 = 'LogFiles/results_combined_27-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search
% [retraining model was not correctly assigned] file_name2 = 'LogFiles/results_combined_27-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search
% file_name1 = 'LogFiles/results_combined_28-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search
% file_name2 = 'LogFiles/results_combined_28-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search
% file_name1 = 'LogFiles/results_combined_29-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
% file_name2 = 'LogFiles/results_combined_29-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
% file_name1 = 'LogFiles/results_combined_30-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
% file_name2 = 'LogFiles/results_combined_30-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
% file_name1 = 'LogFiles/results_combined_31-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
% file_name2 = 'LogFiles/results_combined_31-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
file_name1 = 'LogFiles/results_combined_32-1.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks
file_name2 = 'LogFiles/results_combined_32-2.csv'; % For 01 experiments, 10 new instances with 90% each on old tasks using HyperOpt search and selected new tasks


T1 = readtable(file_name1,'VariableNamingRule','preserve');
T2 = readtable(file_name2,'VariableNamingRule','preserve');

IDs_array = readtable(file_name2,'Range','B2:B' + string(length(T2.Var1)+1)).Var1;
list_latest_new_instance_digit = readtable(file_name2,'Range','C2:C' + string(length(T2.Var1)+1)).Var1;
list_noise_type = readtable(file_name2,'Range','D2:D' + string(length(T2.Var1)+1)).Var1;

acc_digit_0_original_first_model = readtable(file_name2,'Range','J2:J' + string(length(T2.Var1)+1)).Var1;
acc_digit_1_original_first_model = readtable(file_name2,'Range','K2:K' + string(length(T2.Var1)+1)).Var1;
acc_digit_2_original_first_model = readtable(file_name2,'Range','L2:L' + string(length(T2.Var1)+1)).Var1;
acc_digit_3_original_first_model = readtable(file_name2,'Range','M2:M' + string(length(T2.Var1)+1)).Var1;
acc_digit_4_original_first_model = readtable(file_name2,'Range','N2:N' + string(length(T2.Var1)+1)).Var1;
acc_digit_5_original_first_model = readtable(file_name2,'Range','O2:O' + string(length(T2.Var1)+1)).Var1;
acc_digit_6_original_first_model = readtable(file_name2,'Range','P2:P' + string(length(T2.Var1)+1)).Var1;
acc_digit_7_original_first_model = readtable(file_name2,'Range','Q2:Q' + string(length(T2.Var1)+1)).Var1;
acc_digit_8_original_first_model = readtable(file_name2,'Range','R2:R' + string(length(T2.Var1)+1)).Var1;
acc_digit_9_original_first_model = readtable(file_name2,'Range','S2:S' + string(length(T2.Var1)+1)).Var1;

acc_digit_0_original_last_model = readtable(file_name2,'Range','T2:T' + string(length(T2.Var1)+1)).Var1;
acc_digit_1_original_last_model = readtable(file_name2,'Range','U2:U' + string(length(T2.Var1)+1)).Var1;
acc_digit_2_original_last_model = readtable(file_name2,'Range','V2:V' + string(length(T2.Var1)+1)).Var1;
acc_digit_3_original_last_model = readtable(file_name2,'Range','W2:W' + string(length(T2.Var1)+1)).Var1;
acc_digit_4_original_last_model = readtable(file_name2,'Range','X2:X' + string(length(T2.Var1)+1)).Var1;
acc_digit_5_original_last_model = readtable(file_name2,'Range','Y2:Y' + string(length(T2.Var1)+1)).Var1;
acc_digit_6_original_last_model = readtable(file_name2,'Range','Z2:Z' + string(length(T2.Var1)+1)).Var1;
acc_digit_7_original_last_model = readtable(file_name2,'Range','AA2:AA' + string(length(T2.Var1)+1)).Var1;
acc_digit_8_original_last_model = readtable(file_name2,'Range','AB2:AB' + string(length(T2.Var1)+1)).Var1;
acc_digit_9_original_last_model = readtable(file_name2,'Range','AC2:AC' + string(length(T2.Var1)+1)).Var1;

acc_digit_0_noise_0_first_model = readtable(file_name2,'Range','AD2:AD' + string(length(T2.Var1)+1)).Var1;
acc_digit_1_noise_0_first_model = readtable(file_name2,'Range','AE2:AE' + string(length(T2.Var1)+1)).Var1;
acc_digit_2_noise_0_first_model = readtable(file_name2,'Range','AF2:AF' + string(length(T2.Var1)+1)).Var1;
acc_digit_3_noise_0_first_model = readtable(file_name2,'Range','AG2:AG' + string(length(T2.Var1)+1)).Var1;
acc_digit_4_noise_0_first_model = readtable(file_name2,'Range','AH2:AH' + string(length(T2.Var1)+1)).Var1;
acc_digit_5_noise_0_first_model = readtable(file_name2,'Range','AI2:AI' + string(length(T2.Var1)+1)).Var1;
acc_digit_6_noise_0_first_model = readtable(file_name2,'Range','AJ2:AJ' + string(length(T2.Var1)+1)).Var1;
acc_digit_7_noise_0_first_model = readtable(file_name2,'Range','AK2:AK' + string(length(T2.Var1)+1)).Var1;
acc_digit_8_noise_0_first_model = readtable(file_name2,'Range','AL2:AL' + string(length(T2.Var1)+1)).Var1;
acc_digit_9_noise_0_first_model = readtable(file_name2,'Range','AM2:AM' + string(length(T2.Var1)+1)).Var1;

acc_digit_0_noise_0_last_model = readtable(file_name2,'Range','AN2:AN' + string(length(T2.Var1)+1)).Var1;
acc_digit_1_noise_0_last_model = readtable(file_name2,'Range','AO2:AO' + string(length(T2.Var1)+1)).Var1;
acc_digit_2_noise_0_last_model = readtable(file_name2,'Range','AP2:AP' + string(length(T2.Var1)+1)).Var1;
acc_digit_3_noise_0_last_model = readtable(file_name2,'Range','AQ2:AQ' + string(length(T2.Var1)+1)).Var1;
acc_digit_4_noise_0_last_model = readtable(file_name2,'Range','AR2:AR' + string(length(T2.Var1)+1)).Var1;
acc_digit_5_noise_0_last_model = readtable(file_name2,'Range','AS2:AS' + string(length(T2.Var1)+1)).Var1;
acc_digit_6_noise_0_last_model = readtable(file_name2,'Range','AT2:AT' + string(length(T2.Var1)+1)).Var1;
acc_digit_7_noise_0_last_model = readtable(file_name2,'Range','AU2:AU' + string(length(T2.Var1)+1)).Var1;
acc_digit_8_noise_0_last_model = readtable(file_name2,'Range','AV2:AV' + string(length(T2.Var1)+1)).Var1;
acc_digit_9_noise_0_last_model = readtable(file_name2,'Range','AW2:AW' + string(length(T2.Var1)+1)).Var1;

% acc_digit_0_noise_1_first_model = readtable(file_name2,'Range','AX2:AX' + string(length(T2.Var1)+1)).Var1;
% acc_digit_1_noise_1_first_model = readtable(file_name2,'Range','AY2:AY' + string(length(T2.Var1)+1)).Var1;
% acc_digit_2_noise_1_first_model = readtable(file_name2,'Range','AZ2:AZ' + string(length(T2.Var1)+1)).Var1;
% acc_digit_3_noise_1_first_model = readtable(file_name2,'Range','BA2:BA' + string(length(T2.Var1)+1)).Var1;
% acc_digit_4_noise_1_first_model = readtable(file_name2,'Range','BB2:BB' + string(length(T2.Var1)+1)).Var1;
% acc_digit_5_noise_1_first_model = readtable(file_name2,'Range','BC2:BC' + string(length(T2.Var1)+1)).Var1;
% acc_digit_6_noise_1_first_model = readtable(file_name2,'Range','BD2:BD' + string(length(T2.Var1)+1)).Var1;
% acc_digit_7_noise_1_first_model = readtable(file_name2,'Range','BE2:BE' + string(length(T2.Var1)+1)).Var1;
% acc_digit_8_noise_1_first_model = readtable(file_name2,'Range','BF2:BF' + string(length(T2.Var1)+1)).Var1;
% acc_digit_9_noise_1_first_model = readtable(file_name2,'Range','BG2:BG' + string(length(T2.Var1)+1)).Var1;
% 
% acc_digit_0_noise_1_last_model = readtable(file_name2,'Range','BH2:BH' + string(length(T2.Var1)+1)).Var1;
% acc_digit_1_noise_1_last_model = readtable(file_name2,'Range','BI2:BI' + string(length(T2.Var1)+1)).Var1;
% acc_digit_2_noise_1_last_model = readtable(file_name2,'Range','BJ2:BJ' + string(length(T2.Var1)+1)).Var1;
% acc_digit_3_noise_1_last_model = readtable(file_name2,'Range','BK2:BK' + string(length(T2.Var1)+1)).Var1;
% acc_digit_4_noise_1_last_model = readtable(file_name2,'Range','BL2:BL' + string(length(T2.Var1)+1)).Var1;
% acc_digit_5_noise_1_last_model = readtable(file_name2,'Range','BM2:BM' + string(length(T2.Var1)+1)).Var1;
% acc_digit_6_noise_1_last_model = readtable(file_name2,'Range','BN2:BN' + string(length(T2.Var1)+1)).Var1;
% acc_digit_7_noise_1_last_model = readtable(file_name2,'Range','BO2:BO' + string(length(T2.Var1)+1)).Var1;
% acc_digit_8_noise_1_last_model = readtable(file_name2,'Range','BP2:BP' + string(length(T2.Var1)+1)).Var1;
% acc_digit_9_noise_1_last_model = readtable(file_name2,'Range','BQ2:BQ' + string(length(T2.Var1)+1)).Var1;
% 
% acc_digit_0_noise_2_first_model = readtable(file_name2,'Range','BR2:BR' + string(length(T2.Var1)+1)).Var1;
% acc_digit_1_noise_2_first_model = readtable(file_name2,'Range','BS2:BS' + string(length(T2.Var1)+1)).Var1;
% acc_digit_2_noise_2_first_model = readtable(file_name2,'Range','BT2:BT' + string(length(T2.Var1)+1)).Var1;
% acc_digit_3_noise_2_first_model = readtable(file_name2,'Range','BU2:BU' + string(length(T2.Var1)+1)).Var1;
% acc_digit_4_noise_2_first_model = readtable(file_name2,'Range','BV2:BV' + string(length(T2.Var1)+1)).Var1;
% acc_digit_5_noise_2_first_model = readtable(file_name2,'Range','BW2:BW' + string(length(T2.Var1)+1)).Var1;
% acc_digit_6_noise_2_first_model = readtable(file_name2,'Range','BX2:BX' + string(length(T2.Var1)+1)).Var1;
% acc_digit_7_noise_2_first_model = readtable(file_name2,'Range','BY2:BY' + string(length(T2.Var1)+1)).Var1;
% acc_digit_8_noise_2_first_model = readtable(file_name2,'Range','BZ2:BZ' + string(length(T2.Var1)+1)).Var1;
% acc_digit_9_noise_2_first_model = readtable(file_name2,'Range','CA2:CA' + string(length(T2.Var1)+1)).Var1;
% 
% acc_digit_0_noise_2_last_model = readtable(file_name2,'Range','CB2:CB' + string(length(T2.Var1)+1)).Var1;
% acc_digit_1_noise_2_last_model = readtable(file_name2,'Range','CC2:CC' + string(length(T2.Var1)+1)).Var1;
% acc_digit_2_noise_2_last_model = readtable(file_name2,'Range','CD2:CD' + string(length(T2.Var1)+1)).Var1;
% acc_digit_3_noise_2_last_model = readtable(file_name2,'Range','CE2:CE' + string(length(T2.Var1)+1)).Var1;
% acc_digit_4_noise_2_last_model = readtable(file_name2,'Range','CF2:CF' + string(length(T2.Var1)+1)).Var1;
% acc_digit_5_noise_2_last_model = readtable(file_name2,'Range','CG2:CG' + string(length(T2.Var1)+1)).Var1;
% acc_digit_6_noise_2_last_model = readtable(file_name2,'Range','CH2:CH' + string(length(T2.Var1)+1)).Var1;
% acc_digit_7_noise_2_last_model = readtable(file_name2,'Range','CI2:CI' + string(length(T2.Var1)+1)).Var1;
% acc_digit_8_noise_2_last_model = readtable(file_name2,'Range','CJ2:CJ' + string(length(T2.Var1)+1)).Var1;
% acc_digit_9_noise_2_last_model = readtable(file_name2,'Range','CK2:CK' + string(length(T2.Var1)+1)).Var1;

acc_all_digits_original_first_model = readtable(file_name2,'Range','BH2:BH' + string(length(T2.Var1)+1)).Var1;
acc_all_digits_original_last_model = readtable(file_name2,'Range','BI2:BI' + string(length(T2.Var1)+1)).Var1;
acc_all_digits_noise_0_first_model = readtable(file_name2,'Range','BJ2:BJ' + string(length(T2.Var1)+1)).Var1;
acc_all_digits_noise_0_last_model = readtable(file_name2,'Range','BK2:BK' + string(length(T2.Var1)+1)).Var1;
% acc_all_digits_noise_1_first_model = readtable(file_name2,'Range','DT2:DT' + string(length(T2.Var1)+1)).Var1;
% acc_all_digits_noise_1_last_model = readtable(file_name2,'Range','DU2:DU' + string(length(T2.Var1)+1)).Var1;
% acc_all_digits_noise_2_first_model = readtable(file_name2,'Range','DV2:DV' + string(length(T2.Var1)+1)).Var1;
% acc_all_digits_noise_2_last_model = readtable(file_name2,'Range','DW2:DW' + string(length(T2.Var1)+1)).Var1;


% T2 = readtable('LogFiles/results_combined_17-2.csv','VariableNamingRule','preserve');
% T3 = readtable('LogFiles/results_combined_17-3.csv','VariableNamingRule','preserve');

% IDs_array = T1.ID;
Num_of_classes = 10;
FontSize1 = 12;
FontSize2 = 8;

%% Creating plot showing performance change on overall original tasks and new tasks
figure(1)
subplot(1,4,1)
plot(IDs_array,acc_all_digits_original_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_all_digits_original_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
xlabel("Accumilated Retraining ID")
ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('All Digits Original')

subplot(1,4,2)
plot(IDs_array,acc_all_digits_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_all_digits_noise_0_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
xlabel("Accumilated Retraining ID")
ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('All Digits Noise Type 0')

% subplot(1,4,3)
% plot(IDs_array,acc_all_digits_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_all_digits_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('All Digits Noise Type 1')
% 
% subplot(1,4,4)
% plot(IDs_array,acc_all_digits_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_all_digits_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% legend("Location","SouthEast")
% ylim([0 101])
% title('All Digits Noise Type 2')

%% Creating plot showing performance change on each of the original tasks
figure(2)
subplot(1,2,1)
h1 = bar(IDs_array,[T1.acc_instance_test_first_model T1.acc_instance_test_before_last_model T1.acc_instance_test_last_model],'LineWidth',1);hold on;

h1(1).FaceColor = [0.5 0.5 0.5];
h1(1).DisplayName = 'first model';
h1(2).FaceColor = 'w';
h1(2).DisplayName = 'Before last model';
h1(3).FaceColor = 'k';
h1(3).DisplayName = 'Last model';
ylim([0 101])
xlim([-0.5 length(T2.Var1)-0.5])
xlabel("Accumilated Retraining ID")
ylabel("Performance (%)")
title('New Instances')
plot([-0.5 length(T2.Var1)-0.5],[50 50] ,'--r',"LineWidth",2,'DisplayName',"Classification threshold"); hold on;
plot([-0.5 length(T2.Var1)-0.5],[70 70] ,'--g',"LineWidth",2,'DisplayName',"Controlled learning threshold"); hold on;

subplot(1,2,2)
h1 = bar(IDs_array,[T1.acc_accumilated_instances_test_first_model T1.acc_accumilated_instances_test_before_last_model T1.acc_accumilated_instances_test_last_model],'LineWidth',1);hold on;

h1(1).FaceColor = [0.5 0.5 0.5];
h1(1).DisplayName = 'first model';
h1(2).FaceColor = 'w';
h1(2).DisplayName = 'Before last model';
h1(3).FaceColor = 'k';
h1(3).DisplayName = 'Last model';
ylim([0 101])
xlim([-0.5 length(T2.Var1)-0.5])
xlabel("Accumilated Retraining ID")
ylabel("Performance (%)")
title('Accumilated New Instances')
plot([-0.5 9.5],[50 50] ,'--r',"LineWidth",2,'DisplayName',"Classification threshold"); hold on;
plot([-0.5 9.5],[70 70] ,'--g',"LineWidth",2,'DisplayName',"Controlled learning threshold"); hold on;
legend("Location","NorthWest")

% annotation('doublearrow',[0.53 0.97],[0.514 0.514],"Head1Style","none","Head2Style","none","Color","r","LineStyle","--","LineWidth",2)
% annotation('doublearrow',[0.53 0.97],[0.675 0.675],"Head1Style","none","Head2Style","none","Color","g","LineStyle","--","LineWidth",2)
% 
% text(0.2,72,"Controlled learning threshold",'FontSize',FontSize2,"Color","g")
% text(1,52,"Classification threshold",'FontSize',FontSize2,"Color","r")




%% Creating plot (subfigrues) showing  performance change on each of the original tasks
figure(3)
subplot(4,10,1)
plot(IDs_array,acc_digit_0_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_0_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
xlabel("Accumilated Retraining ID")
ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('Original 0')

subplot(4,10,2)
plot(IDs_array,acc_digit_1_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_1_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('1')

subplot(4,10,3)
plot(IDs_array,acc_digit_2_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_2_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('2')

subplot(4,10,4)
plot(IDs_array,acc_digit_3_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_3_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('3')

subplot(4,10,5)
plot(IDs_array,acc_digit_4_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_4_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('4')

subplot(4,10,6)
plot(IDs_array,acc_digit_5_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_5_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('5')

subplot(4,10,7)
plot(IDs_array,acc_digit_6_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_6_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('6')

subplot(4,10,8)
plot(IDs_array,acc_digit_7_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_7_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('7')

subplot(4,10,9)
plot(IDs_array,acc_digit_8_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_8_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('8')

subplot(4,10,10)
plot(IDs_array,acc_digit_9_original_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_9_original_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('9')

%% noise 0
% TODO Make red point dependent on which ID the instance got added to.
subplot(4,10,11)
plot(IDs_array,acc_digit_0_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_0_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
xlabel("Accumilated Retraining ID")
ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('Noise 0')
noise_type_value = 0;
digit = 0;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,12)
plot(IDs_array,acc_digit_1_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_1_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('1')
digit = 1;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,13)
plot(IDs_array,acc_digit_2_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_2_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('2')
digit = 2;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,14)
plot(IDs_array,acc_digit_3_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_3_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('3')
digit = 3;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,15)
plot(IDs_array,acc_digit_4_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_4_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('4')
digit = 4;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,16)
plot(IDs_array,acc_digit_5_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_5_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('5')
digit = 5;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,17)
plot(IDs_array,acc_digit_6_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_6_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('6')
digit = 6;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,18)
plot(IDs_array,acc_digit_7_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_7_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('7')
digit = 7;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,19)
plot(IDs_array,acc_digit_8_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_8_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('8')
digit = 8;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

subplot(4,10,20)
plot(IDs_array,acc_digit_9_noise_0_first_model ,'--r','DisplayName',"Original (first) Model"); hold on;
plot(IDs_array,acc_digit_9_noise_0_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
set(gca,'FontSize',FontSize2)
ylim([0 101])
title('9')
digit = 9;
if any(list_noise_type(:) == noise_type_value)
    indicies_noise = find(list_noise_type==noise_type_value);
    digits_list = list_latest_new_instance_digit(indicies_noise);
    if any(digits_list(:) == digit)
        indicies_digits = find(digits_list(:) == digit);
        for ID_location = indicies_noise(indicies_digits)
            plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
        end
    end
end

% %% noise 1
% subplot(4,10,21)
% plot(IDs_array,acc_digit_0_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_0_noise_1_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('Noise 1')
% digit = 0;
% noise_type_value = 1;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,22)
% plot(IDs_array,acc_digit_1_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_1_noise_1_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('1')
% digit = 1;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,23)
% plot(IDs_array,acc_digit_2_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_2_noise_1_last_model,'-k','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('2')
% digit = 2;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,24)
% plot(IDs_array,acc_digit_3_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_3_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('3')
% digit = 3;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,25)
% plot(IDs_array,acc_digit_4_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_4_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('4')
% digit = 4;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,26)
% plot(IDs_array,acc_digit_5_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_5_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('5')
% digit = 5;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,27)
% plot(IDs_array,acc_digit_6_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_6_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('6')
% digit = 6;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,28)
% plot(IDs_array,acc_digit_7_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_7_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('7')
% digit = 7;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,29)
% plot(IDs_array,acc_digit_8_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_8_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('8')
% digit = 8;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,30)
% plot(IDs_array,acc_digit_9_noise_1_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_9_noise_1_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('9')
% digit = 9;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% %% noise 2
% subplot(4,10,31)
% plot(IDs_array,acc_digit_0_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_0_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% xlabel("Accumilated Retraining ID")
% ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('Noise 2')
% digit = 0;
% noise_type_value = 2;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,32)
% plot(IDs_array,acc_digit_1_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_1_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('1')
% digit = 1;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,33)
% plot(IDs_array,acc_digit_2_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_2_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('2')
% digit = 2;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,34)
% plot(IDs_array,acc_digit_3_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_3_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('3')
% digit = 3;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,35)
% plot(IDs_array,acc_digit_4_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_4_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('4')
% digit = 4;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,36)
% plot(IDs_array,acc_digit_5_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_5_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('5')
% digit = 5;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,37)
% plot(IDs_array,acc_digit_6_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_6_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('6')
% digit = 6;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)acc_digit_9_noise_2_last_model
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,38)
% plot(IDs_array,acc_digit_7_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_7_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('7')
% digit = 7;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,39)
% plot(IDs_array,acc_digit_8_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_8_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('8')
% digit = 8;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end
% 
% subplot(4,10,40)
% plot(IDs_array,acc_digit_9_noise_2_first_model ,'r','DisplayName',"Original (first) Model"); hold on;
% plot(IDs_array,acc_digit_9_noise_2_last_model,'g','DisplayName',"Retrained (latest) Model (EWC + lr) "); hold on;
% % xlabel("Accumilated Retraining ID")
% % ylabel("Performance (%)")
% set(gca,'FontSize',FontSize2)
% ylim([0 101])
% title('9')
% digit = 9;
% if any(list_noise_type(:) == noise_type_value)
%     indicies_noise = find(list_noise_type==noise_type_value);
%     digits_list = list_latest_new_instance_digit(indicies_noise);
%     if any(digits_list(:) == digit)
%         indicies_digits = find(digits_list(:) == digit);
%         for ID_location = indicies_noise(indicies_digits)
%             plot(ID_location,0,'o','MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',3,'DisplayName',"Marker")
%         end
%     end
% end

legend("Location","SouthWest")
% set(gca,'FontSize',FontSize1)
