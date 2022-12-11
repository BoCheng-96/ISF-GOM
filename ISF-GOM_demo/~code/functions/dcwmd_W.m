function [WMD_D] = dcwmd_W(BOW_X,W,Idx_X)
    n  = length(BOW_X);
    WMD_D = zeros(n,n);
    
    maxW = max(max(W));

    parfor i = 1:n
        Ei = zeros(1,n);
        for j = i+1:n
            % fprintf('Computing (%d / %d)\n',i,j);
            if isempty(BOW_X{i}) || isempty(BOW_X{j})
                Ei(j) = 0;
            else
                x1 = BOW_X{i}./sum(BOW_X{i});
                x2 = BOW_X{j}./sum(BOW_X{j});           
                
                C = W(Idx_X{i},Idx_X{j});

                n_i = size(Idx_X{i},2);
                n_j = size(Idx_X{j},2);

                temp_C = reshape(C',1,n_i*n_j);
                temp_idxs = find(temp_C~=maxW);
                [~,I] = sort( temp_C(1,temp_idxs) );
                T = zeros(n_i,n_j);

                nn = length(I);
                sumTC = 0;
                for idx = 1:nn                 
                    nj = mod(temp_idxs(I(idx)),n_j);
                    if nj == 0
                        ni = floor(temp_idxs(I(idx))/n_j);
                        nj = n_j;    
                    else
                        ni = floor(temp_idxs(I(idx))/n_j)+1;
                    end

                    if min(x1(ni),x2(nj)) ~= 0
                        T(ni,nj) = min(x1(ni),x2(nj));
                        x1(ni) = x1(ni) - T(ni,nj);
                        x2(nj) = x2(nj) - T(ni,nj);  

                        sumTC = sumTC + T(ni,nj) * C(ni,nj);
                    end
                end
                Ei(j) = sumTC + (1 - sum(sum(T)))*maxW;
            end
        end 
        WMD_D(i,:) = Ei;
    end    
end

