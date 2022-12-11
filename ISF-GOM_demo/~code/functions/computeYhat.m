function [idx_sw,Yhat] = computeYhat(Idx_seedwords,W,alpha,k,num_sw,isPropagation)
    v = size(W,2);
    l = length(Idx_seedwords);
    
    Yhat = zeros(v,l);
    idx_sw = zeros(num_sw,l);
%     I = 1:v;
       
    for i = 1:l 
        idx_sw(:,i) = Idx_seedwords{i}(1:num_sw); 
        Yhat(idx_sw(:,i),i) = 1;  % initial Yhat       
    end
%       Yhat(setdiff(I,reshape(idx_sw,num_sw*l,1)),:) = 1/l;
   

    if isPropagation == 1
        P = Yhat;
        p0 = P;                                                     %Ô­Ê¼µÃpartial label
        iterVal = zeros(1,1000);  

        for i = 1:v
            [~,I] = sort(W(i,:));
            W(i,I(1:v-k)) = 0;
            W(i,:) = W(i,:)./sum(W(i,:));
        end    

        W = (W+W')./2; 

        sumW = full(sum(W,2));
        sumW(sumW==0)=1;
        W = bsxfun(@rdivide,W,sumW);        
        
        for iter=1:5
            tmp = P; 
        %        P = alpha*W*P+(1-alpha)*p0;                    %propagate Y<-T
            P = alpha*W*P+(1-alpha)*P;                    %propagate Y<-T
            rows = find(any(P,2)==1);
            P(rows,:) = P(rows,:)./repmat(sum(P(rows,:),2),1,l);  

            for i = 1:l
                P(idx_sw(:,i),:) = 0;
                P(idx_sw(:,i),i) = 1;  % reset seedwords' weights
            end

            diff=norm(full(tmp)-full(P),2);

            iterVal(iter) = abs(diff);
            if abs(diff)<0.00001
                break
            end
        end
        Yhat = P;    
        clear P        
    end
end