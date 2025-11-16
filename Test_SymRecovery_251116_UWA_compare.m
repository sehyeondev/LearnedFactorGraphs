% StaSPNet example code - ISI channel with AWGN
clear all;
close all;
clc;

rng(1);

%% Parameters setting
s_nConst = 2;       % Constellation size (2 = BPSK)
s_nMemSize = 4;     % Number of taps
s_fTrainSize = 500; % Training size
s_fTestSize = 5000; % Test data size
s_nRxAntennas = 2;

s_nStates = s_nConst^s_nMemSize;

v_fSigWdB= -6:2:10;  %Noise variance in dB

s_fEstErrVar = 0.1;   % Estimation error variance
% Frame size for generating noisy training
s_fFrameSize = 50;
s_fNumFrames = s_fTrainSize/s_fFrameSize;

v_nCurves   = [...          % Curves
    1 ...                   % StaSPNet  - EKF
    1 ....                  % StaSPNet - BKNet
    1 ...                   % SP algorithm
    ];


s_nCurves = length(v_nCurves);

v_stProts = strvcat(  ...
    'BCJRNet, EKF', ...
    'BCJRNet, BKNet',...
    'BCJR, GTD');

% Network parameters
NetParams.DNN = 3; % 3 layers
NetParams.maxEpochs = 100;
NetParams.DropPeriod = 100;
NetParams.miniBatchSize = 27;
NetParams.learnRate = 0.01;

%% Simulation loop
v_fExps =  0.1:0.1:0.1;
m_fSERAvg = zeros(length(v_nCurves),length(v_fSigWdB));

load('channel_data_SNR20dB.mat')
load('channel_data_EKF_SNR20dB.mat')
load('channel_data_BKNet_SNR20dB.mat')


%%
numTimes = 100;
numBatches = 10;
for timeIdx=1:numTimes
for batchIdx=1:numBatches
    % Exponentailly decaying channel
    % m_fChannel =  repmat(exp(-v_fExps(eIdx)*(0:(s_nMemSize-1))), s_nRxAntennas, 1);
    % m_fChannel = m_fChannelKNet;

    m_fChannelGTD = reshape(H_f_GTD(batchIdx,:,timeIdx), 4, 2)';
    m_fChannelEKF = reshape(H_f_EKF(batchIdx,:,timeIdx), 4, 2)';
    m_fChannelKNet = reshape(H_f_KNet(batchIdx,:,timeIdx), 4, 2)';
    
    
    m_fSER = zeros(length(v_nCurves),length(v_fSigWdB));
    
    
    % Generate training labels
    v_fXtrain = randi(s_nConst,1,s_fTrainSize);
    v_fStrain = 2*(v_fXtrain - 0.5*(s_nConst+1));
    m_fStrain = m_fMyReshape(v_fStrain, s_nMemSize);
    
    % Training with perfect CSI
    m_Rtrain = fliplr(m_fChannelGTD) * m_fStrain;

    % Training with noisy CSI - EKF
    m_Rtrain = zeros(size(m_Rtrain));
    for kk=1:s_fNumFrames
        Idxs=((kk-1)*s_fFrameSize + 1):kk*s_fFrameSize;
        m_Rtrain(:,Idxs) =  fliplr(m_fChannelEKF + sqrt(s_fEstErrVar)*randn(size(m_fChannelEKF)).*m_fChannelEKF) ...
           * m_fStrain(:,Idxs);
    end
    
    % Training with noisy CSI - BKNet
    m_Rtrain2 = zeros(size(m_Rtrain));
    for kk=1:s_fNumFrames
        Idxs=((kk-1)*s_fFrameSize + 1):kk*s_fFrameSize;
        m_Rtrain2(:,Idxs) =  fliplr(m_fChannelKNet + sqrt(s_fEstErrVar)*randn(size(m_fChannelKNet)).*m_fChannelKNet) ...
           * m_fStrain(:,Idxs);
    end
    
    
    % Generate test labels
    v_fXtest = randi(s_nConst,1,s_fTestSize);
    v_fStest = 2*(v_fXtest - 0.5*(s_nConst+1));
    m_fStest= m_fMyReshape(v_fStest, s_nMemSize);
    m_Rtest = fliplr(m_fChannelGTD) * m_fStest;
    
    
    
    % Get state tranisition matrix
    m_fTransition = m_fTransMat(s_nConst, s_nMemSize, v_fXtrain);
    
    % Loop over number of SNR
    for mm=1:length(v_fSigWdB)
        s_fSigmaW = 10^(-0.1*v_fSigWdB(mm));
        % LTI AWGN channel
        m_fYtrain = m_Rtrain + sqrt(s_fSigmaW)*randn(size(m_Rtrain));
        m_fYtrain2 = m_Rtrain2 + sqrt(s_fSigmaW)*randn(size(m_Rtrain));
        m_fYtest = m_Rtest + sqrt(s_fSigmaW)*randn(size(m_Rtest));
        
        tic;
        % StaSPNet - EKF
        if(v_nCurves(1)==1)
            % Train network
            [net, GMModel] = GetSPNet(v_fXtrain, m_fYtrain ,s_nConst, s_nMemSize, NetParams);
            % Apply StaSPNet detctor
            [~, v_fXhat] =  ApplySPNet(m_fYtest, net, GMModel, s_nConst, m_fTransition); 
            % Evaluate error rate
            m_fSER(1,mm) = mean(v_fXhat ~= v_fXtest);
        end
        
        % StaSPNet - BKNet
        if(v_nCurves(2)==1)
            % Train network using training with uncertainty
            [net, GMModel] = GetSPNet(v_fXtrain, m_fYtrain2 ,s_nConst, s_nMemSize, NetParams);
            % Apply StaSPNet detctor
            [~, v_fXhat] =  ApplySPNet(m_fYtest, net, GMModel, s_nConst, m_fTransition); 
            % Evaluate error rate
            m_fSER(2,mm) = mean(v_fXhat ~= v_fXtest);
            
        end
        
        % Model-based SP algorithm
        if(v_nCurves(3)==1)
            
            m_fLikelihood = zeros(s_fTestSize,s_nStates);
            % Compute coditional PDF for each state
            for ii=1:s_nStates
                v_fX = zeros(s_nMemSize,1);
                Idx = ii - 1;
                for ll=1:s_nMemSize
                    v_fX(ll) = mod(Idx,s_nConst) + 1;
                    Idx = floor(Idx/s_nConst);
                end
                v_fS = 2*(v_fX - 0.5*(s_nConst+1));
                m_fLikelihood(:,ii) = mvnpdf(bsxfun(@minus,m_fYtest,fliplr(m_fChannelGTD)*v_fS)',zeros(1,2),s_fSigmaW*eye(2)); 
                % m_fLikelihood(:,ii) = normpdf(m_fYtest'-fliplr(m_fChannel)*v_fS,0,s_fSigmaW);
            end
            % Apply SP detection based on computed likelihoods
            v_fXhat = v_fSumProduct(m_fLikelihood, s_nConst, m_fTransition);
            % Evaluate error rate
            m_fSER(3,mm) = mean(v_fXhat ~= v_fXtest);
        end
        toc;
        % Dispaly exponent index
        % Display SNR index
        batchIdx
        mm
    end
    m_fSERAvg = m_fSERAvg + m_fSER;
end
end
m_fSERAvg = m_fSERAvg/(numBatches*numTimes);


%% Display results
v_stPlotType = strvcat( '-rs', '--bo', '-kx', '--mv', '-.b^');

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


