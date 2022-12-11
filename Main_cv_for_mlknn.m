clear all
clc
format compact

% Perform (our method: GOM, ISF-GOM) for mlknn task.
addpath(genpath('~mat'))
addpath(genpath('~code'))
addpath(genpath('MLKNN'))

datasetNames = ["eurlex" "delicious_multi" "bibtex" "delicious" "tmc2007_10000" "bookmarks_10000"];
wmd_Styles = ["dcwmd"];
% C_Styles = [0 1];
C_Styles = [0];
% Value_Style = ["Mean" "Max"];
Value_Style = ["Mean"];

for dataset = datasetNames
    % load data
    dataPath = ['~mat/data_cv/' char(dataset) '_new' '.mat'];
    load(dataPath);

    % Create file to save results
    fid=fopen('~Results/' + string(dataset)+ '_' + string(wmd_Styles) + '_' + string(Value_Style) + '_'+ string(C_Styles) + '_Result.txt','wt');
    fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\n', ... 
        'Dataset','numFold','Wmd_style','C_style','Value_Style', 'k', 'hamming_loss','one_error','average_precision','ratio','mu');
    clear WE
    
    v = length(word_list);

    fprintf('Compute W ...\n');  
    load(['~mat/Cosine_W/' char(dataset) '_CosW' '.mat']);
    % Compute similarities between words
    %     norm_WE = sqrt(sum(abs(WE).^2));
%     W = zeros(v,v);
%     for i = 1:v
%         for j = i+1:v
%             W(i,j) = (1 - WE(:,i)' * WE(:,j)/ (norm_WE(i) * norm_WE(j)))/2;
%         end
%     end
%     save(['~mat/Cosine_W/' char(dataset) '_CosW' '.mat'], 'W')
    % -----------------------------------------------------------------------   

    W = W + W';
    sortW = sort(reshape(W,1,v*v));
    for ratio = [1e-5 5*1e-5 1e-4 5*1e-4 0.001 0.005 0.01 0.05 0.1 0.5 1]
        ids = (floor(v*(v-1)*ratio)+v);
        mu = sortW(1,ids);
        
        new_W = W;
        switch Value_Style
            case "Mean"
                new_W(new_W>mu) = mean(mean(W));
            case "Max"
                new_W(new_W>mu) = max(max(W));
        end

        for wmd_style = wmd_Styles    
            for cc = C_Styles
                fprintf('%s: %s, ratio=%f, mu=%f \n',dataset,wmd_style,ratio,mu); 
                
                switch cc
                    case 0
                        c_style = "";                  
                    case 1
                        c_style = "_W";      
                end

                % -------------------- Compute distances between documents -----------------------------------
                % call ~code/functions/funcname
                funName = wmd_style + c_style;
                fprintf('Compute WMD ...\n');
                switch cc
                    case 0
                        % Don't replace W with new_W                                        
                        WMD_D = feval(char(funName),BOW_X,Idx_X);
                        end  
                    case 1
                        % Replace W with new_W
                        WMD_D = feval(char(funName),BOW_X,new_W,Idx_X);                     
                end

                % WMD_D is a upper triangular matrix
                temp_WMD = WMD_D + WMD_D';

                % -------------------- Compute metrics with different k and fold for MLKNN task ! -----------------------------------
                fprintf('Compute Err ...\n');
                nfold = size(TR,1);
                ks = [1:100];
                Smooth = 1;

                sum_hamming_loss = 0;
                sum_one_error = 0;
                sum_aps = 0;
                for i = 1:nfold
                   fprintf('Start at fold:%d \n ----- ',i)
                   Train_idx = TR(i,:);
                   Test_idx = TE(i,:);
                   
                   % train_data: An MxN array, the ith instance of training instance is stored in train_data(i,:)
                   % train_target: A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
                   ytr = Y(Train_idx, :);
                   train_target = ytr';
                   train_target(train_target==0) = -1;
                   yte = Y(Test_idx, :);
                   test_target = yte';
                   test_target(test_target==0) = -1;

                   DE = temp_WMD(Test_idx,Train_idx);
                   DT = temp_WMD(Train_idx,Train_idx);
                   
                   hamming_loss = ones(1, length(ks));
                   one_error = ones(1, length(ks));
                   aps = ones(1, length(ks));
                   for k = 1:length(ks)
                       [Prior, PriorN, Cond, CondN] = MLKNN_train(DT, train_target, k, Smooth);
                       [HammingLoss, OneError, Average_Precision, Outputs, Pre_Labels] = MLKNN_test(DE, train_target, test_target, k, Prior, PriorN, Cond, CondN);
                       hamming_loss(1, k) = HammingLoss;
                       one_error(1, k) = OneError;
                       aps(1, k) = Average_Precision;
                   end
                   sum_hamming_loss = sum_hamming_loss + hamming_loss;
                   sum_one_error = sum_one_error + one_error;
                   sum_aps = sum_aps + aps;  
                end           
                
                % Average metrics with nfold!
                final_hamming_loss = sum_hamming_loss / nfold; 
                final_one_error = sum_one_error / nfold;
                final_aps = sum_aps / nfold;
                
                % Output and Save minimum metrics(chosen in ks)! 
                for k = 1:length(ks)
                    fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %d\t %f\t %f\t %f\t %f\t %f\n', ...
                    dataset, "cv-"+string(nfold), wmd_style, c_style, Value_Style, ks(k), final_hamming_loss(k), final_one_error(k), final_aps(k), ratio, mu);
                end
%                 fprintf('final_hamming_loss:%g\n', final_hamming_loss);
%                 [B1,I1] = sort(final_hamming_loss);
%                 fprintf('Minimum_hamming_loss:%g in k = %d\n', B1(1), ks(I1(1)));
%                 fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %s\t %d\t %f\t %f\t %f\n', ...
%                     dataset, "cv-"+string(nfold), wmd_style, c_style, Value_Style, "hamming_loss", ks(I1(1)), B1(1), ratio, mu);
%                 fprintf('final_one_error:%g\n', final_one_error);
%                 [B2,I2] = sort(final_one_error);
%                 fprintf('Minimum_one_error:%g in k = %d\n', B2(1), ks(I2(1)));
%                 fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %s\t %d\t %f\t %f\t %f\n', ...
%                     dataset, "cv-"+string(nfold), wmd_style, c_style, Value_Style, "one_error", ks(I2(1)), B2(1), ratio, mu);
%                 fprintf('final_aps:%g\n', final_aps);
%                 [B3,I3] = sort(final_aps, 'descend');
%                 fprintf('Maximum_average_precision:%g in k = %d\n', B3(1), ks(I3(1)));
%                 fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %s\t %d\t %f\t %f\t %f\n', ...
%                     dataset, "cv-"+string(nfold), wmd_style, c_style, Value_Style, "average_precision", ks(I3(1)), B3(1), ratio, mu);
                % -----------------------------------------------------------------------
            end
        end
    end
    
    fclose(fid);
end