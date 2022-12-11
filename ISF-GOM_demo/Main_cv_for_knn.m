clear all
format compact

% Perform (our method: GOM, ISF-GOM) for knn task.
addpath('emd')
addpath(genpath('~mat'))
addpath(genpath('~code'))

% single-label datasets
datasetNames = ["stackoverflow" "biomedical" "twitter" "snippets" "r8" "classic" "amazon" "recipe2" "ohsumed" "20ng" "webkb" "bbcsport"];

wmd_Styles = ["dcwmd"];

% C_Styles = [0 1];
C_Styles = [0];

% Value_Styles = ["Mean" "Max"]
Value_Style = ["Mean"];

for dataset = datasetNames
    % load data
    dataPath = ['~mat/data_cv/' char(dataset) '_new' '.mat'];
    load(dataPath);  
    
    v = size(WE,2);    
    clear WE
    
    fid=fopen('~Results/' + string(dataset)+ '_' + string(wmd_Styles) + '_' + string(Value_Style) + '_Result.txt','wt');
    fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\n', ... 
        'Dataset','numFold','Wmd_style','C_style','Value_Styles','k','Err','ratio','mu');     
    

    % -------------------- Compute W -----------------------------------
    
%     fprintf('Compute W ...\n'); 
    load(['~mat/Cosine_W/' char(dataset) '_CosW' '.mat']);
    
    if dataset == "20ng"
        W = [W1;W2];
        clear W1 W2
    end      

%     norm_WE = sqrt(sum(abs(WE).^2));
%     W = zeros(v,v);
%     for i = 1:v
%         for j = i+1:v
%             W(i,j) = (1 - WE(:,i)' * WE(:,j)/ (norm_WE(i) * norm_WE(j)))/2;
%         end
%     end
% %     W1 = W(1:10500,:);
% %     W2 = W(10501:21000,:);
%     
%     save(['~mat/Cosine_W/' char(dataset) '_CosW' '.mat'], 'W') 
% -----------------------------------------------------------------------   
    
    W = W + W';
    mean_W = mean(mean(W));
    max_W = max(max(W));
    
    [sortW,~] = sort(reshape(W,1,v*v));

    for ratio = [1e-5 5*1e-5 1e-4 5*1e-4 0.001 0.005 0.01 0.05 0.1 0.5 1]
        ids = (floor(v*(v-1) * ratio) + v);
        mu = sortW(1,ids);
        new_W = W;

        switch Value_Style
            case "Mean"
                new_W(new_W>mu) = mean_W;
            case "Max"
                new_W(new_W>mu) = max_W;
        end            

        for wmd_style = wmd_Styles    
            for cc = C_Styles
                fprintf('%s: %s, ratio=%f, mu=%f, mean_W=%f \n',dataset,wmd_style,ratio,mu,mean_W); 
                
                switch cc
                    case 0
                        c_style = "";                  
                    case 1
                        c_style = "_W";      
                end

                funName = wmd_style + c_style;
                fprintf('Compute WMD ...\n');
                switch cc
                    case 0
                        % Without ISF                                         
                        WMD_D = feval(char(funName),BOW_X,Idx_X);
                        end  
                    case 1
                        % With ISF
                        WMD_D = feval(char(funName),BOW_X,new_W,Idx_X);
                end
                WMD_D = WMD_D + WMD_D';

% ----------------------------------------------------------------------- 
                fprintf('Compute Err ...\n');
                nfold = size(TR,1);
                ks = [1:19];    

                sum_err = 0;
                for i = 1:nfold
                   Train_idx = TR(i,:);
                   Test_idx = TE(i,:);

                   ytr = Y(1,Train_idx);
                   yte = Y(1,Test_idx);

                   DE = WMD_D(Train_idx,Test_idx);

                   sum_err = sum_err + knn_fall_back(DE,ytr,yte,ks);  
                end           

                final_err = sum_err/nfold;
                [B,I] = sort(final_err);
                fprintf('%s --- Minimum_err:%g in k = %d\n', dataset, B(1), ks(I(1)));
                fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %d\t %f\t %f\t %f\n', ...
                    dataset, "cv-"+string(nfold), wmd_style, c_style, Value_Style, ks(I(1)), B(1), ratio, mu);
            end
        end
        
        clear WMD_D
    end
    
    fclose(fid);
end