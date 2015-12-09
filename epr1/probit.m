%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2015
%
% probit.m
%
%
% calculates the data log likelihood of a probit.  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ logZhatOUT, ZhatOUT , muHatOUT , sigmaHatOUT , HOUT ] = probit( m , K , parms ) 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % some useful parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % C should be 1 in the univariate case, and v should be the magnitude
    % in the multidimensional case, we compile them as they should be.
    % note that, to accord with other ep, with assume C is unit length in
    % its column vectors.  thus v carries the magnitude.
    C = [parms.C];
    v = [parms.v];
        
    errorCheck = 0; % cleaner, but requires a bit more computation...
    n = length(m);
    p = size(C,2);
    % number of sample points to use with sampling method
    numPoints = 100000;
    
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
    
    if n == 1 && p == 1 && ~isnan(v)
        % using rasmussen and williams 2006 sec 3.9
        % using that terminology: m is mu, v is v, K is sig2
        % then 1 d problem... can be solved in closed form
        v = 1/v;
        % recall v is a singleton
        z = ( m )/(v*sqrt(1+(K/v^2)));
        npz = normpdf(z);
        ncz = normcdf(z);
        
        logZhatOUT = log(ncz);
        
        ZhatOUT = exp(logZhatOUT);
        % in the 1d case, these matter a lot for EP
        muHatOUT  = m + (K * npz)/(ncz*(v*sqrt(1+(K/v^2))));
        sigmaHatOUT = K - ((K^2*npz)/((v^2 + K)*ncz))*( z + (npz)/(ncz));
        HOUT = nan;

        if 1
            % calculate entropy with sampling...
            % also let's put these random points somewhere else
            z = randn( n , numPoints );
            %x = bsxfun( @plus , sqrt(K)*z , m );
            x = bsxfun( @plus , sqrt(sigmaHatOUT)*z , muHatOUT ); % importance sampler
            logqcav = log(normpdf(x , m , sqrt(K)));
            logti = log(normcdf( x/v , 0 , 1 ));

            logw = log(normpdf(x , muHatOUT , sqrt(sigmaHatOUT) ));
            %HOUT = logZhatOUT - nanmean( exp( logti - logZhatOUT ).*(logqcav + logti ) );
            HOUT = - nanmean( exp( logqcav + logti - logw - logZhatOUT ).*(logqcav + logti - logZhatOUT) );
            
            % check Zhat
            %logZhatcheck = log(mean(exp(logti + logqcav - logw)));
            %muHatcheck = mean( exp(logti + logqcav - logw - logZhatOUT).*x);
            %sigmaHatcheck = mean( exp(logti + logqcav - logw - logZhatOUT).*x.^2) - muHatcheck.^2;
            %Hnorm = ( 1/2*log(2*pi*exp(1)) + 1/2*log(sigmaHatOUT) );
            %if HOUT > (Hnorm + .01)
            %    keyboard
            %end
            %a = [logZhatOUT logZhatcheck ; muHatOUT muHatcheck ; sigmaHatOUT sigmaHatcheck ; HOUT Hnorm];
            %[ a diff(a,1,2) ]
            
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
            logpPt = logpPt + log( normcdf(  v(j)*C(:,j)'*x  , 0 , 1 ) );
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

    
    
    