%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2015
%
% plot H results
%%%%%%%%%%%%%%%%%

function [ results ] = plot_results_H( saveFigs , results , lik , scenario , j , test_name )

    %%%%%%%%%%%%
    % check inputs
    %%%%%%%%%%%%
    if nargin < 6 || isempty(j)
        test_name = 'no test_name given';
    end
    if nargin < 5 || isempty(j)
        j = 4;
    end
    if nargin < 4 || isempty(scenario)
        scenario = 1;
    end
    if nargin < 3 || isempty(lik)
        lik = @probit;
    end
    if nargin < 2 || isempty(results)
        load(sprintf('results/results_%s_%d.mat',func2str(lik),scenario));
    end
    if nargin < 1 || isempty(saveFigs)
        saveFigs = 0;
    end

    %%%%%%%%%%%%
    % preprocess
    %%%%%%%%%%%%
    plotpts = zeros(size(results));
    for i = 1 : size(results,1)
            plotx(i,1) = results(i,j).p;
            % true H
            ploty(i,1) = results(i,j).Htrue;
            results(1,1).test_line_name = '$$H(p)$$';
            % H of q ep
            ploty(i,2) = ( results(i).n/2*log(2*pi*exp(1)) + 1/2*logdet(results(i,j).Sigmaep) );
            results(1,2).test_line_name = '$$H(q_{ep})$$';
            % H of ep
            ploty(i,3) = ploty(i,2) + sum( results(i,j).extrasep.fracTerms.*(results(i,j).extrasep.Htilted - results(i,j).extrasep.Hnorm)); 
            results(1,3).test_line_name = '$$H_{ep}$$';
            % H of tilted
            %ploty(i,4) = mean( results(i,j).extrasep.Htilted ); 
            %results(1,4).test_line_name = 'H(tilt)';
            % H of moment match
            ploty(i,4) = ( results(i).n/2*log(2*pi*exp(1)) + 1/2*logdet(results(i,j).sigmatrue) );
            results(1,4).test_line_name = '$$H(q_{mm})$$';
            % error in logZ
            ploty(i,5) =  results(i,j).logZtrue - results(i,j).logZep ; 
            results(1,5).test_line_name = '$$\log Z - \log Z_{ep}$$';
            % error in H
            ploty(i,6) =  ploty(i,1) - ploty(i,3); 
            results(1,6).test_line_name = '$$H(p) - H_{ep}$$';

    end
    
    %%%%%%%%%%%%
    % figure
    %%%%%%%%%%%%
    figure('position',[1 600, 600, 500]);
    fs = 20;
    lw = 3;
    %col = [ 27,158,119 ; 217,95,2 ; 117,112,179 ; 231,41,138 ; 102,166,30 ; 230,171,2 ]'/255;
    col = [ 228,26,28 ; 55,126,184 ; 77,175,74 ; 255,127,0 ; 152,78,163 ; 247,129,191]'/255;
    %set(gca,'fontsize',fs);
    axes('XScale','log','YScale','linear');
    xlabel('Number of Factors  ','fontsize',fs);
    xlim([1 1e4]);
    %ylim([-0.1 1.1]);
    ylabel('H','fontsize',fs);
    hold on
    % plot some fiducial markers
    plot([1 1e4],-0.0*[1 1],'k--','linewidth',1);
    %plot([1 1e4],-0.1*[1 1],'k--','linewidth',lw);
    %plot([1 1e4],-1*[1 1],'k--','linewidth',lw);
    % plot each case
    for k = 1 : size(ploty,2)
        semilogx( plotx(:,1) , ploty(:,k) , 'color', col(:,k) , 'linewidth',lw);
        % annotate line
        plotstr = results(1,k).test_line_name;
        text( plotx(end,1)*1.2 , ploty(end,k) ,  plotstr , 'color', col(:,k) , 'fontsize', fs ,'interpreter', 'latex');
    end
    % set fonts after all plots to avoid plotting weirdness
    set(gca, 'fontsize',fs);
    title(sprintf('%s %s',test_name, func2str(lik)),'interpreter','latex');

        
    if saveFigs
        if isequal(test_name,'$$\infty$$')
            test_name = 'infty';
        end
        figname = sprintf('figs/results_%s_%s_%d_H',func2str(lik),test_name,scenario);
        %print(gcf,'-depsc',sprintf('figs/%s.eps',figname));
        print(gcf,'-depsc',sprintf('%s.eps',figname));
        saveTightFigure( gcf , sprintf('%s.pdf',figname) );
    end
end


