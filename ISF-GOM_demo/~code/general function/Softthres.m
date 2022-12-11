function P = Softthres(P,lambda)
    P = max(P-lambda,0) - max(-P-lambda,0); 
end