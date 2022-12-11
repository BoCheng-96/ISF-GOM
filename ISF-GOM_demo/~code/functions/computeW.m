function [W] = computeW(WE,lambda1,lambda2,lambda3)
% ---------------------------------------------------------------------------------
% Calculate the weight matrix for each sample reconstructed from the other samples 
% X = WE : dim(v*d)
% W      : dim(v*v)
% X, X_hat, R, J : dim(v*d)
% W, S : dim(v*v)
% ---------------------------------------------------------------------------------

    X = WE'; % dim(v*d)
    [v,d] = size(X);
    
    % initial settings
    X_hat = X;
    R = zeros(v,d);
    W = rand(v,v).*(ones(v,v)-eye(v));
    S = W;
    J = X_hat;
    theta1 = zeros(v,d);
    theta2 = zeros(v,v);
    theta3 = zeros(v,d);
    tao1 = 1;
    tao2 = 1;
    tao3 = 1; 
    
%     lambda1 = 0.01;
%     lambda2 = 0.1;
%     lambda3 = 0.1;
    
    tol=1e-4;

    MaxIter = 50;
    display_iter = 1;

    tic;
    t1 = clock;
    for iter = 1:MaxIter
    
        % -----------------------------------------------------------------
        % 1 Update W,S,theta2
        % 1-1 Update S
        tempJ = J*J';
        S = (tempJ + (tao2.*W - theta2)./2) * (tempJ + (tao2/2).*eye(v))^-1;

        % 1-2 Update W
        W = Softthres(S+theta2./tao2, lambda1/tao2);
        W = W - eye(v)*1e3;
        W(W<0) = 0;
        W = W./repmat(sum(W,2),1,v); % 加和为1，归一化
%         W = (W - repmat(min(W,[],2),1,v)) ./ (repmat(max(W,[],2)-min(W,[],2),1,v));
        
        
        fprintf('Nonzero Number: %d\n',length(find(W(1,:)~=0)));

        % 1-3 Updata theta2
        theta2 = theta2 + tao2.*(S-W);

        % -----------------------------------------------------------------
        % Update R
        R = (1/(tao1+2*lambda3)).*(X-J+theta1./tao1);

        % -----------------------------------------------------------------
        % Update X_hat, J, theta1, theta3
        % Update J
        tempS = eye(v)-S;
        J = (2.*tempS*tempS' + (tao1+tao3).*eye(v))^-1 * (tao1.*(X-R) + theta1 + tao3.*X_hat - theta3);

        % Updata X_hat
        [U,sigma,V] = svd(J+theta3./tao3);
        X_hat = U * Softthres(sigma, lambda2/tao3) * V';

        % Update theta1 and theta3
        theta1 = theta1 + tao1.*(X-J-R);
        theta3 = theta3 + tao3.*(J-X_hat);  
        
        tempXW = (eye(v)-W)*X_hat;
        obj1 = trace(tempXW'*tempXW);
        obj2 = lambda3 * trace(R'*R);
        obj(iter) = obj1 + obj2;       
        
        if (iter>2 && ((norm((obj(iter)-obj(iter-1)),'fro'))^2/(norm(obj(iter-1),'fro'))^2)<tol)
            break;
        end

        if (iter==1 || mod(iter, display_iter)==0)  
            fprintf('Iter %d, Obj: %g, Time:%g\n', iter, obj(iter), etime(clock,t1));
        end        
    end
        
end

