%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2015
%
% gmmLL.m
%
% calculates the necessary epr1 calcs with a gmm likelihood.  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ logZhatOUT, ZhatOUT , muHatOUT , sigmaHatOUT , HOUT ] = gmmLL( m , K , parms ) 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % some useful parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % C should be 1 in the univariate case.
    % note that, to accord with other ep, with assume C is unit length in
    % its column vectors.  
    C = [parms.C];
    % the remaining parameters are the means, variances, and weights of a
    % gaussian mixture observation:
    % parms(j).w ... weights vector
    % parms(j).mu(k) ... vector of 1d means... remember this is for rank one EP, so
    % by definition this is a one dimensional GMM, so k indexes over the
    % number of mixture components
    % parms(j).y(k) ... the implicit observation. 
    % parms(j).sigma(k) ... vector of 1d variances... as above.
    
    
    errorCheck = 0; % cleaner, but requires a bit more computation...
    n = length(m);
    p = size(C,2);
    % number of sample points to use with sampling method
    numPoints = 500000;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % first some error checking
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if errorCheck
        % check sizes
        if n~=size(C,1) || n~=size(K,1) || n~=size(K,2)
            fprintf('ERROR: the mean vector is not the same size as the columns of C or K (or K is not square).\n');
            keyboard
        end
        % check Sigma
        if norm(K - K')>1e-14;
            fprintf('ERROR: your covariance matrix is not symmetric.\n');
            keyboard;
        end
    end   
    
    if n == 1 && p == 1 
        
        % here is the 1d problem that we can solve in closed form
        % this is a parms.w weighted set of normal distributions.
        
        % build the components...
        sigprod = zeros(length(parms.w),1);
        muprod = zeros(length(parms.w),1);
        Zinvprod = zeros(length(parms.w),1);
        for k = 1 : length(parms.w)
            % see for example rasmussen's book, appendix A.8
            sigprod(k) = 1/(1/K + 1/parms.sigma(k));
            muprod(k) = sigprod(k)*( m/K + (parms.mu(k) - parms.y)/parms.sigma(k) );
            Zinvprod(k) = 1/(sqrt(2*pi)*sqrt(K + parms.sigma(k)))*exp( -1/2*( m - (parms.mu(k) - parms.y) )*inv(K + parms.sigma(k))*(m-(parms.mu(k)-parms.y)) );
        end
        % see notes p189
        ZhatOUT = sum( parms.w .* Zinvprod );
        logZhatOUT = log(ZhatOUT);
        muHatOUT = 1/ZhatOUT .* sum( parms.w .* Zinvprod .* muprod );
        sigmaHatOUT = 1/ZhatOUT .* sum( parms.w .* Zinvprod .* ( sigprod + muprod.^2) ) - muHatOUT^2;
        % entropy is numerical...
        HOUT = nan;
      
        if 1
            % calculate entropy with sampling...
            % also let's put these random points somewhere else
            z = randn( n , numPoints );
            %x = bsxfun( @plus , sqrt(K)*z , m );
            x = bsxfun( @plus , sqrt(sigmaHatOUT)*z , muHatOUT ); % importance sampler
            % now the product terms... first the given cavity...
            logqcav = log(normpdf(x , m , sqrt(K)));
            % now the actual gmm
            
            % the only part specific to this likelihood... 
            logti = log( pdf( gmdistribution( (parms.mu - parms.y) , reshape(parms.sigma,[1 1 length(parms.mu)])  , parms.w' ) , x' ) )';

            % the importance sampler weights.
            logw = log(normpdf(x , muHatOUT , sqrt(sigmaHatOUT) ));
            %HOUT = logZhatOUT - nanmean( exp( logti - logZhatOUT ).*(logqcav + logti ) );
            HOUT = - nanmean( exp( logqcav + logti - logw - logZhatOUT ).*(logqcav + logti - logZhatOUT) );
            
            % check Zhat
%             logZhatcheck = log(mean(exp(logti + logqcav - logw)));
%             muHatcheck = mean( exp(logti + logqcav - logw - logZhatOUT).*x);
%             sigmaHatcheck = mean( exp(logti + logqcav - logw - logZhatOUT).*x.^2) - muHatcheck.^2;
%             Hnorm = ( 1/2*log(2*pi*exp(1)) + 1/2*log(sigmaHatOUT) );
%             if HOUT > (Hnorm + .01)
%                 keyboard
%             end
%             a = [logZhatOUT logZhatcheck ; muHatOUT muHatcheck ; sigmaHatOUT sigmaHatcheck ; HOUT Hnorm];
%             [ a diff(a,1,2) ]
            
        end
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % sample prior points... really weak sampling approach, but it should
        % serve in low dimensions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        R = chol(K);
        z = randn( n , numPoints );
        x = bsxfun( @plus , R'*z , m );
        % now calc the sample likelihood
        %logmgp = -log(numPoints);
        logpPt = zeros(1,numPoints);
        for j = 1 : p
            logpPt = logpPt + log( pdf( gmdistribution( (parms(j).mu - parms(j).y) , reshape(parms(j).sigma,[1 1 length(parms(j).mu)])  , parms(j).w' ) , (C(:,j)'*x)' ) )';
        end
        logZhatOUT = log(mean(exp( logpPt )));
        ZhatOUT = exp(logZhatOUT);
        % mean and covariance (for entropy and other questions)
        muHatOUT  = mean( bsxfun(@times , x , exp(logpPt) ),2)/ZhatOUT;
        % just 1d case for sanity checking.
        % x is n by numPoints.  x*x' is n by n, sum of outerproducts. 
        % x*diag(exp(logpPt))*x' likewise does the same
        %sigmaHatOUT = 1/numPoints*(x*diag(exp(logpPt))*x')/ZhatOUT - muHatOUT*muHatOUT';
        sigmaHatOUT = 1/numPoints*(bsxfun(@times, x , exp(logpPt))*x')/ZhatOUT - muHatOUT*muHatOUT';
        %sigmaHatOUTcheck = (mean( bsxfun(@times , x.^2 , exp(logpPt) )))/ZhatOUT - muHatOUT^2;
        
        % H out
        logpx = -n/2*log( 2*pi ) - 1/2*logdet(K) - 1/2*sum(z.*z,1);
        % not quite right, as nans here get inappropriately counted...
        % recall that in entropy we define 0 log 0 = 0, so we prioritize
        % exp logpPt - logZhatOUT as a zero.
        use_inds = isfinite(logpPt);
        % now sum over all good values, and normalize by the TOTAL number
        HOUT = sum( exp( logpPt(use_inds) - logZhatOUT ).*( logZhatOUT - logpPt(use_inds) - logpx(use_inds) ))/numPoints;
        %HOUT  = nanmean( exp( logpPt - logZhatOUT ).*( logZhatOUT - logpPt - logpx ));
        %keyboard
    end
end

    
    
    