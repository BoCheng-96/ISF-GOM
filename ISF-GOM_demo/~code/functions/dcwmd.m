function [WMD_D] = dcwmd(BOW_X,X)
    n  = length(BOW_X);
    WMD_D = zeros(n,n);

    parfor i = 1:n
        Ei = zeros(1,n);
        for j = i+1:n
            if isempty(BOW_X{i}) || isempty(BOW_X{j})
                Ei(j) = 0;
            else
                x1 = BOW_X{i}./sum(BOW_X{i});
                x2 = BOW_X{j}./sum(BOW_X{j});         
                
                C = sqrt(max(distance(X{i},X{j}),0));

                n_i = size(X{i},2);
                n_j = size(X{j},2);
                [~,I] = sort( reshape( C',1,n_i*n_j) );
                T = zeros(n_i,n_j);
                nn = length(find(C~=1));
                for idx = 1:n_i*n_j                   
                    nj = mod(I(idx),n_j);
                    if nj == 0
                        ni = floor(I(idx)/n_j);
                        nj = n_j;    
                    else
                        ni = floor(I(idx)/n_j)+1;
                    end
                    
                    if min(x1(ni),x2(nj)) == 0
                        T(ni,nj) = 0;
                    else
                        T(ni,nj) = min(x1(ni),x2(nj));
                        x1(ni) = x1(ni) - T(ni,nj);
                        x2(nj) = x2(nj) - T(ni,nj);        
                    end
                end
                Ei(j) = sum(sum(T.*C));
            end
        end
        WMD_D(i,:) = Ei;
    end
end

