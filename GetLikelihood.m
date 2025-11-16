function m_fLikelihood = GetLikelihood(m_fYtest,m_fChannel,s_nStates,s_nMemSize,nVar,M)
% Compute coditional PDF for each state
m_fLikelihood = zeros(length(m_fYtest), s_nStates);
for ii=1:s_nStates
    v_fX = zeros(s_nMemSize,1);
    Idx = ii - 1;
    for ll=1:s_nMemSize
        v_fX(ll) = mod(Idx,M) + 1;
        Idx = floor(Idx/M);
    end
    v_fS = 2*(v_fX - 0.5*(M+1));
    % m_fLikelihood(:,ii) = mvnpdf(bsxfun(@minus,m_fYtest,fliplr(m_fChannel)*v_fS)',zeros(1,2),s_fSigmaW*eye(2));
    m_fLikelihood(:,ii) = normpdf(m_fYtest'-fliplr(m_fChannel)*v_fS,0,nVar);
end