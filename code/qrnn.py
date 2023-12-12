import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

class QRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits = 4,
        n_qlayers = 2,
        batch_first=True,
        backend = "default.qubit"
    ):
        
        super(QRNN, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        self.batch_first = batch_first

        self.wires = [f"wire_{i}" for i in range(self.n_qubits)]
        self.dev = qml.device(self.backend, wires = self.wires)

        def _layer_qrnn_block(W):
            qml.RX(W[0], wires = 'wire_0')
            qml.RZ(W[0], wires = 'wire_0')
            qml.RX(W[0], wires = 'wire_0')

            qml.RX(W[1], wires = 'wire_1')
            qml.RZ(W[1], wires = 'wire_1')
            qml.RX(W[1], wires = 'wire_1')

            qml.RX(W[2], wires = 'wire_2')
            qml.RZ(W[2], wires = 'wire_2')
            qml.RX(W[2], wires = 'wire_2')

            qml.RX(W[3], wires = 'wire_3')
            qml.RZ(W[3], wires = 'wire_3')
            qml.RX(W[3], wires = 'wire_3')

            qml.CNOT(wires = ['wire_0','wire_1'])
            qml.RZ(W[1], wires = 'wire_1')
            qml.CNOT(wires = ['wire_0','wire_1'])

            qml.CNOT(wires = ['wire_1','wire_2'])
            qml.RZ(W[2], wires = 'wire_2')
            qml.CNOT(wires = ['wire_1','wire_2'])

            qml.CNOT(wires = ['wire_2','wire_3'])
            qml.RZ(W[3], wires = 'wire_3')
            qml.CNOT(wires = ['wire_2','wire_3'])

            qml.CNOT(wires = ['wire_3', 'wire_0'])
            qml.RZ(W[0], wires = 'wire_0')
            qml.CNOT(wires = ['wire_3','wire_0'])


        def _circuit_qrnn_block(inputs, weights):
            qml.AngleEmbedding(inputs, self.wires)

            for W in weights:
                _layer_qrnn_block(W)

            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires]
        
        self.qlayer_circuit = qml.QNode(_circuit_qrnn_block, self.dev, interface = "torch")

        weights_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'circuit': qml.qnn.TorchLayer(self.qlayer_circuit, weights_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states = None):
        
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            x_t = x[:, t, :]

            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)

            h_t = torch.sigmoid(self.clayer_out(self.VQC['circuit'](y_t)))
            # o_t = torch.sigmoid(self.clayer_out(self.VQC(y_t)))

            # h_t = torch.tanh(o_t)

        return h_t
