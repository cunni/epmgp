%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2015
% 
% test_epr1.m
% lik is a handle to the moment calculator, as in probitLL, gmmLL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ results ] = test_epr1( lik , scenario )

    % input check
    if nargin < 2 || isempty(scenario)
        scenario = 1;
    end
    if nargin < 1 || isempty(lik)
        lik = @probit;
    end
    % setup features of the tests that are invariant to scenario
    % number of points forming each trace in the graph
    num_test_pts = 10;
    largest_test_pt = 2^num_test_pts;
    % number of lines
    test_line_names = {'weak', 'medium', 'strong','$$\infty$$','random'};
    % number of factors
    factors_list = round(2.^linspace(0,log2(largest_test_pt),num_test_pts))';
   
    switch scenario
        % simple 1d, without any alpha correction (here's the problem)
        case {1,2,4}
            % dimension of the problem
            dim_list = scenario*ones(num_test_pts,1);
            % prior specification
            for i = 1 : num_test_pts
                m_list{i} = zeros(dim_list(i),1);
                if scenario==1
                    K_list{i} = eye(dim_list(i));
                elseif scenario==2
                    K_list{i} = [1 0.5 ; 0.5 1];
                else 
                    K_list{i} = 0.75*eye(4) + 0.25*ones(4);
                end
            end
            % now generate the random regions
            for j = 1 : length(test_line_names)
                % switch on the likelihood type
                switch func2str(lik)
                    case 'probit'
                        % check for the random one
                        if isequal(test_line_names{j},'random')
                            % then generate C, the observations, from a probit
                            % model.
                            for i = 1 : num_test_pts
                                % something like what is observed in real bayesian
                                % probit regression.
                                C_list{i,j} = 40*rand( dim_list(i) , factors_list(i) ) - .1;
                                % or draw from probit model
                                % first draw the function shape 'x' (the betas)
                                x = chol(K_list{i})'*randn(dim_list(i),1) + m_list{i};
                                C_list{i,j} = 10*(rand( dim_list(i) , factors_list(i)))+2;
                                %C_list{i,j} = (ones( dim_list(i) , factors_list(i)));
                                %C_list{i,j} = 1/sqrt(2)*(ones( dim_list(i) , factors_list(i)));
                                % now draw points conditioned on these draws... the
                                % C vectors thus fare are only z, they need to be
                                % signed by y_i|x,z_i ~ bern( normcdf( x'z_i ))
                                % \in {-1,1}
                                yy = 2*(rand(1,factors_list(i)) <= x'*C_list{i,j})-1;
                                C_list{i,j} = bsxfun(@times, C_list{i,j} , yy );
                                % normalize C and assign its magnitude to v
                                v_list{i,j} = sqrt(sum(C_list{i,j}.*C_list{i,j},1));
                                C_list{i,j} = C_list{i,j}./repmat(v_list{i,j},dim_list(i),1);
                                parms = [];
                                for k = 1 : factors_list(i)
                                    % note that C really should be made unit norm...
                                    parms(k).C = C_list{i,j}(:,k);
                                    parms(k).v = v_list{i,j}(k);
                                end
                                p_list{i,j} = parms;
                            end
                        else
                            % pick out the numerical value corresponding to
                            % this name.
                            tl_names = {'weak', 'medium', 'strong','$$\infty$$'};
                            tl = {1, 2 , 14 , 500 };
                            test_line_val = tl{ find( strcmp(tl_names,test_line_names{j})) };
                            for i = 1 : num_test_pts
                                % when there are multiple dimensions, we want to
                                % spread out evenly the factors across those
                                % dimensions.  If a perfect multiple, this is eye
                                for k = 1 : factors_list(i)
                                    % add each factor one by one
                                    % first we make a unit vector ek
                                    ek = zeros(dim_list(i),1);
                                    ek( mod( k , dim_list(i) ) + 1 ) = 1;
                                    C_list{i,j}(:,k) = test_line_val*ek;
                                    %C_list{i,j} = test_lines{j}*repmat(eye(dim_list(i)), 1, factors_list(i));
                                end
                                % normalize C and assign its magnitude to v
                                v_list{i,j} = sqrt(sum(C_list{i,j}.*C_list{i,j},1));
                                C_list{i,j} = C_list{i,j}./repmat(v_list{i,j},dim_list(i),1);
                                parms = [];
                                for k = 1 : factors_list(i)
                                    % note that C really should be made unit norm...
                                    parms(k).C = C_list{i,j}(:,k);
                                    parms(k).v = v_list{i,j}(k);
                                end
                                p_list{i,j} = parms;
                            end
                        end
            
                    %gmm
                    case 'gmm'
                        if isequal(test_line_names{j},'random')
                            % then generate C directions at random
                            for i = 1 : num_test_pts
                                C_list{i,j} = randn( dim_list(i) , factors_list(i) );
                                % normalize C and assign its magnitude to v
                                v_list{i,j} = sqrt(sum(C_list{i,j}.*C_list{i,j},1));
                                C_list{i,j} = C_list{i,j}./repmat(v_list{i,j},dim_list(i),1);
                                % first draw the gmm location 'x' (the central location)
                                x = chol(K_list{i})'*randn(dim_list(i),1) + m_list{i};
                                % we make each GMM have two components
                                parms = [];
                                for k = 1 : factors_list(i)
                                    parms(k).C = C_list{i,j}(:,k);
                                    parms(k).w(1,1) = rand;
                                    parms(k).w(2,1) = 1 - parms(k).w(1,1);
                                    parms(k).mu = randn(2,1);
                                    parms(k).sigma = .1*rand(2,1);
                                    % draw a data point from each
                                    parms(k).y = random( gmdistribution( (parms(k).mu - C_list{i,j}(:,k)'*x) , reshape(parms(k).sigma,[1 1 length(parms(k).mu)])  , parms(k).w' ));  
                                end
                                p_list{i,j} = parms;
                                % now we have w, mu, sigma, y
                            end
                        else
                            % pick out the numerical value corresponding to
                            % this name.
                            tl_names = {'weak', 'medium', 'strong','$$\infty$$'};
                            tl = {1, 2 , 14 , 500 };
                            test_line_val = tl{ find( strcmp(tl_names,test_line_names{j})) };
                            for i = 1 : num_test_pts
                                % when there are multiple dimensions, we want to
                                % spread out evenly the factors across those
                                % dimensions.  If a perfect multiple, this is eye
                                for k = 1 : factors_list(i)
                                    % add each factor one by one
                                    % first we make a unit vector ek
                                    ek = zeros(dim_list(i),1);
                                    ek( mod( k , dim_list(i) ) + 1 ) = 1;
                                    C_list{i,j}(:,k) = test_line_val*ek;
                                    %C_list{i,j} = test_lines{j}*repmat(eye(dim_list(i)), 1, factors_list(i));
                                end
                                % normalize C and assign its magnitude to v
                                v_list{i,j} = sqrt(sum(C_list{i,j}.*C_list{i,j},1));
                                C_list{i,j} = C_list{i,j}./repmat(v_list{i,j},dim_list(i),1);
                            
                                % first draw the gmm location 'x' (the central location)
                                x = chol(K_list{i})'*randn(dim_list(i),1) + m_list{i};
                                % now we choose the gmm to be of different
                                % strength of nongaussianity.
                                parms = [];
                                for k = 1 : factors_list(i)
                                    parms(k).C = C_list{i,j}(:,k);
                                    parms(k).w(1,1) = rand;
                                    parms(k).w(2,1) = 1 - parms(k).w(1,1);
                                    parms(k).mu = [1 ; -1];
                                    parms(k).sigma = 1/test_line_val*ones(2,1);
                                    % draw a data point from each
                                    parms(k).y = random( gmdistribution( (parms(k).mu - C_list{i,j}(:,k)'*x) , reshape(parms(k).sigma,[1 1 length(parms(k).mu)])  , parms(k).w' ));
                                end
                                p_list{i,j} = parms;
                                % now we have w, mu, sigma, y
                            end
                        end
                    otherwise
                        fprintf('Bad likelihood specification!\n');
                        keyboard
                end
                        
            end    
            
            % now generate the alphas
            for j = 1 : length(test_line_names)
                for i = 1 : num_test_pts
                    alpha{i,j} = 1./(factors_list(i)).^(2/3);                    
                    alpha{i,j} = 1;
                end
            end
            
            
        otherwise
            fprintf('not yet!');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % now run the ep algorithm and its numerical comparison.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1 : length(test_line_names)
        for i = 1 : length(dim_list)
            % set ep_parms
            ep_parms = [];
            for k = 1: size(C_list{i,j},2)
                ep_parms(k).alphaCorrection = alpha{i,j};
            end
            % run it.
            fprintf('scenario %d, dim %d, factors %d, iter %d of %d, case %d of %d.\n',scenario,dim_list(i),factors_list(i),i,length(dim_list),j,length(test_line_names));
            % ep call
            [results(i,j).logZep , results(i,j).muep , results(i,j).Sigmaep, results(i,j).extrasep] = epr1( m_list{i} , K_list{i} , C_list{i,j} , lik , p_list{i,j} , ep_parms );

            % true moment call
            [ results(i,j).logZtrue, results(i,j).Ztrue , results(i,j).mutrue , results(i,j).sigmatrue, results(i,j).Htrue ] = feval( lik, m_list{i} , K_list{i} , p_list{i,j} );

            %[ results(i,j).Ztrue results(i,j).Htrue ]
            % write results
            results(i,j).m = m_list{i};
            results(i,j).K = K_list{i};
            results(i,j).C = C_list{i,j};
            results(i,j).parms = p_list{i,j};
            %results(i,j).v = v_list{i,j};
            results(i,j).p = size(C_list{i,j},2);
            results(i,j).n = dim_list(i);
            %results(i,j).test_line = test_lines{j}; % strength of obs
            results(i,j).test_line_name = test_line_names{j}; % name for that
            
            % write some accuracy results
            results(i,j).logZdiff = (results(i,j).logZep - results(i,j).logZtrue);
            results(i,j).logZnormeddiff = (results(i,j).logZep - results(i,j).logZtrue)/abs(results(i,j).logZtrue);
            %results(i,j).logZnormeddiff = ((results(i,j).logZep + sum([results(i,j).extrasep.Hnorm - results(i,j).extrasep.Htilted])) - results(i,j).logZtrue)/abs(results(i,j).logZtrue);
            results(i,j).Hdiff = sum([results(i,j).extrasep.Hnorm - results(i,j).extrasep.Htilted]);
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%
        % save test_line temp results
        %%%%%%%%%%%%%%%%%%%%%%%
        savename = sprintf('results/results_%s_%d_tmp.mat',func2str(lik),scenario);
        save(savename,'results');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % save final results 
    %%%%%%%%%%%%%%%%%%%%%%%
    savename = sprintf('results/results_%s_%d.mat',func2str(lik),scenario);
    save(savename,'results');
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % plot results 
    %%%%%%%%%%%%%%%%%%%%%%%
    % plot Z results
    plot_results_Z( 1 , results , lik , scenario );
    % now plot entropies
    for i = 1 : length(test_line_names)
        plot_results_H( 1 , results , lik , scenario , i , test_line_names{i});
    end
            