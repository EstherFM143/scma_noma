% SCMA Simulation: BER vs SNR(dB)

% Define SCMA Codebooks
CB(:,:,1) = [ 0                  0                  0                  0;...
             -0.1815-1j*0.1318  -0.6351-1j*0.4615   0.6351+1j*0.4615   0.1815+1j*0.1318;...
              0                  0                  0                  0;...
              0.7851            -0.2243             0.2243            -0.7851 ];

CB(:,:,2) = [ 0.7851            -0.2243             0.2243            -0.7851;...
              0                  0                  0                  0;...
             -0.1815-1j*0.1318  -0.6351-1j*0.4615   0.6351+1j*0.4615   0.1815+1j*0.1318;...
              0                  0                  0                  0 ];

CB(:,:,3) = [-0.6351+1j*0.4615   0.1815-1j*0.1318  -0.1815+1j*0.1318   0.6351-1j*0.4615;...
              0.1392-1j*0.1759   0.4873-1j*0.6156  -0.4873+1j*0.6156  -0.1392+1j*0.1759;...
              0                  0                  0                  0;...
              0                  0                  0                  0 ];

CB(:,:,4) = [ 0                  0                  0                  0;...
              0                  0                  0                  0;...
              0.7851            -0.2243             0.2243            -0.7851;...
             -0.0055-1j*0.2242  -0.0193-1j*0.7848   0.0193+1j*0.7848   0.0055+1j*0.2242 ];

CB(:,:,5) = [-0.0055-1j*0.2242  -0.0193-1j*0.7848   0.0193+1j*0.7848   0.0055+1j*0.2242;...
              0                  0                  0                  0;...
              0                  0                  0                  0;...
             -0.6351+1j*0.4615   0.1815-1j*0.1318  -0.1815+1j*0.1318   0.6351-1j*0.4615 ];

CB(:,:,6) = [ 0                  0                  0                  0;...
              0.7851            -0.2243             0.2243            -0.7851;...
              0.1392-1j*0.1759   0.4873-1j*0.6156  -0.4873+1j*0.6156  -0.1392+1j*0.1759;...
              0                  0                  0                  0 ];

% Parameters
K = size(CB, 1); % orthogonal resources
M = size(CB, 2); % codewords per codebook
V = size(CB, 3); % users (layers)

N = 10000; % SCMA symbols per frame

EbN0 = 0:20;
SNR  = EbN0 + 10*log10(log2(M)*V/K);

Nerr  = zeros(V, length(SNR));
Nbits = zeros(V, length(SNR));
BER   = zeros(V, length(SNR));

maxNumErrs = 100;
maxNumBits = 1e7;
Niter      = 10;

for k = 1:length(SNR)
    N0 = 1/(10^(SNR(k)/10)); % noise power

    while ((min(Nerr(:,k)) < maxNumErrs) && (Nbits(1,k) < maxNumBits))

        % Generate random symbols for each user
        x = randi([0 M-1], V, N); 

        % Rayleigh fading channel
        h = 1/sqrt(2)*(randn(K, V, N)+1j*randn(K, V, N));

        % SCMA encoding
        s = scmaenc(x, CB, h); 
        y = awgn(s, SNR(k));

        % SCMA decoding (MPA algorithm)
        LLR = scmadec(y, CB, h, N0, Niter);

        % Symbol to bits
        r    = de2bi(x, log2(M), 'left-msb');
        data = zeros(log2(M)*N, V);
        for kk = 1:V
            data(:,kk) = reshape(downsample(r, V, kk-1).',[],1);
        end

        % LLR to bits
        datadec = reshape((LLR <= 0), [log2(M) N*V]).';
        datar   = zeros(log2(M)*N, V);
        for kk = 1:V
            datar(:,kk) = reshape(downsample(datadec, V, kk-1).', [], 1);
        end

        % Count errors
        err        = sum(xor(data, datar));
        Nerr(:,k)  = Nerr(:,k) + err.';
        Nbits(:,k) = Nbits(:,k) + log2(M)*N;
    end
    BER(:,k) = Nerr(:,k)./Nbits(:,k);
end

% Plot results
figure;
semilogy(EbN0, BER(1,:), 'b-', 'DisplayName', 'User 1'); hold on;
semilogy(EbN0, BER(2,:), 'r-', 'DisplayName', 'User 2');
semilogy(EbN0, BER(3,:), 'y-', 'DisplayName', 'User 3');
semilogy(EbN0, BER(4,:), 'm-', 'DisplayName', 'User 4');
semilogy(EbN0, BER(5,:), 'g-', 'DisplayName', 'User 5');
semilogy(EbN0, BER(6,:), 'c-', 'DisplayName', 'User 6');
grid on;
xlabel('Eb/N0 (dB)'); ylabel('BER');
legend('show');
title('BER vs Eb/N0 performance for six SCMA users');

