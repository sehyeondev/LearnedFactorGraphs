m_fPriors = m_fLikelihood;
s_nMemSize = round(log(size(m_fTransition,1)) / log(s_nConst));
s_nDataSize = size(m_fPriors, 1);
s_nStates = s_nConst^s_nMemSize;
v_fShat = zeros(1, s_nDataSize);

% Generate state switch matrix - each state appears exactly Const times
m_fStateSwitch = zeros(s_nStates,s_nConst);
for ii=1:s_nStates
    Idx = floor((ii -1)/s_nConst) + 1;
    for ll=1:s_nConst
        m_fStateSwitch(ii,ll) = (s_nStates/s_nConst)*(ll-1) + Idx;
    end
    
end

% Compute forward messages path
m_fForward = zeros(s_nStates, 1+s_nDataSize);
% assume that the initial state is only zero (state 1)
m_fForward(1,1) = 1;
for kk=1:s_nDataSize
   for ii=1:s_nStates 
       for ll=1:s_nConst
           s_nNextState = m_fStateSwitch(ii,ll);
           m_fForward(s_nNextState, kk+1) = m_fForward(s_nNextState, kk+1) + ...
                                            m_fForward(ii,kk)*m_fPriors(kk,s_nNextState)...
                                            *m_fTransition(s_nNextState,ii);   
       end
   end
   % Normalize
    m_fForward(:, kk+1) =  m_fForward(:, kk+1) / sum( m_fForward(:, kk+1));
end