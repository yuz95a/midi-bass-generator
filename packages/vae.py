class MIDIBassVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MIDIBassVAE, self).__init__()
        
        # 인코더
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # 디코더
        self.decoder = nn.LSTM(latent_dim + input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        _, (h, _) = self.encoder(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, other_tracks):
        # z: [batch_size, latent_dim]
        # other_tracks: [batch_size, seq_len, input_dim]
        
        batch_size, seq_len, _ = other_tracks.shape
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        decoder_input = torch.cat([z_expanded, other_tracks], dim=2)
        
        output, _ = self.decoder(decoder_input)
        return self.output_layer(output)
        
    def forward(self, other_tracks):
        mu, log_var = self.encode(other_tracks)
        z = self.reparameterize(mu, log_var)
        bass_track = self.decode(z, other_tracks)
        return bass_track, mu, log_var