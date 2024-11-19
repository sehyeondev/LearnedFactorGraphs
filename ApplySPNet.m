function [v_fXhat, v_fXhat2] = ApplySPNet(v_fY, net, GMModel, s_nConst, m_fTransition)

% Apply SPNet to observations
%
% Syntax
% -------------------------------------------------------
% [v_fXhat, v_fXhat2] =  ApplySPNet(v_fY, net, GMModel, s_nConst, m_fTransition)
% INPUT:
% -------------------------------------------------------
% v_fY - channel output vector
% net - trained neural network model
% GMModel - trained mixture model PDF estimate
% s_nConst - constellation size (positive integer)
% m_fTransition - state transition matrix
% 
%
% OUTPUT:
% -------------------------------------------------------
% v_fXhat - recovered symbols vector using direct application
% v_fXhat2 - recovered symbols vector using StaSPNet 
 
s_nStates = size(m_fTransition,1);

% Use network to compute likelihood function 
m_fpS_Y = predict(net, num2cell(v_fY,1)); 

% Apply argmax to softmax output
[~, v_fXhat] = max(m_fpS_Y');

% Compute output PDF
v_fpY = pdf(GMModel, v_fY');
% Compute likelihoods
m_fLikelihood = (m_fpS_Y .* v_fpY)*s_nStates;
% Apply sum product
v_fXhat2 = v_fSumProduct(m_fLikelihood, s_nConst, m_fTransition);
% v_fXhat2 = v_fSumProduct(m_fpS_Y, s_nConst, m_fTransition);