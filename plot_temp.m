v_stPlotType = strvcat( '-bs', '--bx', '-.k^', '--k+', '-rsquare', '--r*', '-.gv', '--gdiamond');

v_stLegend = [];
fig1 = figure;
set(fig1, 'WindowStyle', 'docked');
%
for aa=1:s_nCurves
    if (v_nCurves(aa) ~= 0)
        v_stLegend = strvcat(v_stLegend,  v_stProts(aa,:));
        semilogy(v_fSigWdB, m_fSERAvg(aa,:), v_stPlotType(aa,:),'LineWidth',1,'MarkerSize',10);
        hold on;
    end
end

xlabel('SNR [dB]');
ylabel('Symbol error rate');
grid on;
legend(v_stLegend,'Location','SouthWest');
hold off;