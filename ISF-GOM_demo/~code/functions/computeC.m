function [C] = computeC(Yhat,W,idx_sw,isPropagation)
    v = size(Yhat,1);    
    [num_sw,l] = size(idx_sw);
    
    Yhat(Yhat==0) = 1e-7;
    
    if isPropagation == 1
        E = zeros(v,1);
        
        for i = 1:v
            E(i,1) = exp((-sum(Yhat(i,:).*log(Yhat(i,:)))));        
        end
        EE = E * E';
        r = zeros(v,v);
        
        vec_idx_sw = reshape(idx_sw,1,num_sw*l);

        W = 1 - (W+W')./2 - eye(size(W,1));
        C = W;
        tempW = zeros(v,v);
        for i = 1:v
            for j = vec_idx_sw
                r(i,j) = sqrt(sum((Yhat(i,:) - Yhat(j,:)).^2));
                C(i,j) = r(i,j) * W(i,j);
                C(j,i) = r(i,j) * W(i,j);
%                 C(i,j) = EE(i,j) * r(i,j) * W(i,j);
            end
        end
    else    
        I = 1:v;
        vec_idx_sw = reshape(idx_sw,1,num_sw*l);
        for i = vec_idx_sw
            E(i,1) = exp((-sum(Yhat(i,:).*log(Yhat(i,:)))));        
        end
        EE = E * E';
        r = zeros(v,v);

        W = 1 - (W+W')./2 - eye(size(W,1));
        C = W;
        for i = vec_idx_sw
            for j = vec_idx_sw
                r(i,j) = sqrt(sum((Yhat(i,:) - Yhat(j,:)).^2));
                C(i,j) = EE(i,j) * r(i,j) * W(i,j);
            end
        end
    end   
    
end