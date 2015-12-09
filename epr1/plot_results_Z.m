%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2015
%
% plot plot_results_Z
%%%%%%%%%%%%%%%%%

function [ results ] = plot_results_Z( saveFigs , results , lik , scenario )

    %%%%%%%%%%%%
    % check inputs
    %%%%%%%%%%%%
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
        for j = 1 : size(results,2)
            plotx(i,j) = results(i,j).p;
            %if plotH
            %    ploty(i,j) = -results(i,j).Hdiff./results(i,j).Htrue;
            %else
            ploty(i,j) = results(i,j).logZnormeddiff;
            % for plotting corrections
            %ploty(i,j) = (ploty(i,j)*abs(results(i,j).logZtrue) + ( results(i,j).Htrue - ( results(i).n/2*log(2*pi*exp(1)) + 1/2*logdet(results(i,j).Sigmaep) + sum( results(i,j).extrasep.fracTerms.*(results(i,j).extrasep.Htilted - results(i,j).extrasep.Hnorm)) ) ))/abs(results(i,j).logZtrue);
            %end
        end
    end
    
    %%%%%%%%%%%%
    % figure
    %%%%%%%%%%%%
    figure('position',[1 600, 600, 500]);
    fs = 20;
    lw = 3;
    %col = [ 27,158,119 ; 217,95,2 ; 117,112,179 ; 231,41,138 ; 102,166,30 ; 230,171,2 ]'/255;
    %col = [ 228,26,28 ; 55,126,184 ; 77,175,74 ; 152,78,163 ; 255,127,0 ; 255,255,51]'/255;
    col = [ 127,205,187 ; 65,182,196 ; 29,145,192 ; 34,94,168 ; 37,52,148 ; 8,29,88 ]'/255;
    %set(gca,'fontsize',fs);
    axes('XScale','log','YScale','linear');
    xlabel('Number of Factors  ','fontsize',fs);
    %xlim([1 2e3]);
    %ylim([-2.5 .5]);
    xlim([1 1e4]);
    ylabel('Normalized error in logZ','fontsize',fs);
    hold on
    % plot some fiducial markers
    plot([1 1e4],-0.0*[1 1],'k--','linewidth',1);

    % plot each case
    for j = 1 : size(results,2)
        semilogx( plotx(:,j) , ploty(:,j) , 'color', col(:,j) , 'linewidth',lw);
        % annotate line
        plotstr = results(1,j).test_line_name;
        text( plotx(end,j)*1.2 , ploty(end,j) ,  plotstr , 'color', col(:,j) , 'fontsize', fs ,'interpreter', 'latex');
    end
    % set fonts after all plots to avoid plotting weirdness
    set(gca, 'fontsize',fs);
    title(func2str(lik),'interpreter','latex');

    if saveFigs
        figname = sprintf('figs/results_%s_%d_Z',func2str(lik),scenario);
        print(gcf,'-depsc',sprintf('%s.eps',figname));
        saveTightFigure( gcf , sprintf('%s.pdf',figname) );
    end
end




