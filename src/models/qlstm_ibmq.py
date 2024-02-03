import pennylane as qml
import torch
import torch.nn as nn


class QLSTM_IBMQ(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits,
                 n_qlayers,
                 variational_layer,
                 batch_first=True):
        super(QLSTM_IBMQ, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers

        self.batch_first = batch_first
        self.backend = "ibm_kyoto"
        self.ibmq_token = "XXX"

        self.dev = qml.device('qiskit.ibmq', wires=n_qubits, backend=self.backend, ibmqx_token=self.ibmq_token)

        self.wires_forget = range(64)
        self.wires_input = range(64)
        self.wires_update = range(64)
        self.wires_output = range(64)

        self.dev_forget = qml.device('qiskit.ibmq', wires=self.wires_forget, backend=self.backend, ibmqx_token=self.ibmq_token)
        self.dev_input = qml.device('qiskit.ibmq', wires=self.wires_input, backend=self.backend, ibmqx_token=self.ibmq_token)
        self.dev_update = qml.device('qiskit.ibmq', wires=self.wires_update, backend=self.backend, ibmqx_token=self.ibmq_token)
        self.dev_output = qml.device('qiskit.ibmq', wires=self.wires_output, backend=self.backend, ibmqx_token=self.ibmq_token)

        def _circuit_forget(inputs, weights):
            num_slices = 16
            slice_size = 4
            num_qubits = 127

            # Loop through each slice to apply AngleEmbedding and variational_layer
            for slice_idx in range(num_slices):
                start_qubit = (slice_idx * slice_size) % num_qubits
                # Map numerical indices to custom wire labels
                wires_slice = range(start_qubit, start_qubit + slice_size)

                qml.templates.AngleEmbedding(inputs[start_qubit:start_qubit + slice_size], wires=wires_slice)
                qml.templates.BasicEntanglerLayers(weights, wires=wires_slice)

            # After encoding all slices, measure all used qubits at once
            all_expvals = [qml.expval(qml.PauliZ(wires=w)) for w in range(num_slices * slice_size)]
            return all_expvals

        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            num_slices = 16
            slice_size = 4
            num_qubits = 127

            # Loop through each slice to apply AngleEmbedding and variational_layer
            for slice_idx in range(num_slices):
                start_qubit = (slice_idx * slice_size) % num_qubits
                # Map numerical indices to custom wire labels
                wires_slice = range(start_qubit, start_qubit + slice_size)

                qml.templates.AngleEmbedding(inputs[start_qubit:start_qubit + slice_size], wires=wires_slice)
                qml.templates.BasicEntanglerLayers(weights, wires=wires_slice)

            # After encoding all slices, measure all used qubits at once
            all_expvals = [qml.expval(qml.PauliZ(wires=w)) for w in range(num_slices * slice_size)]
            return all_expvals

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            num_slices = 16
            slice_size = 4
            num_qubits = 127

            # Loop through each slice to apply AngleEmbedding and variational_layer
            for slice_idx in range(num_slices):
                start_qubit = (slice_idx * slice_size) % num_qubits
                # Map numerical indices to custom wire labels
                wires_slice = range(start_qubit, start_qubit + slice_size)

                qml.templates.AngleEmbedding(inputs[start_qubit:start_qubit + slice_size], wires=wires_slice)
                qml.templates.BasicEntanglerLayers(weights, wires=wires_slice)

            # After encoding all slices, measure all used qubits at once
            all_expvals = [qml.expval(qml.PauliZ(wires=w)) for w in range(num_slices * slice_size)]
            return all_expvals

        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            num_slices = 16
            slice_size = 4
            num_qubits = 127

            # Loop through each slice to apply AngleEmbedding and variational_layer
            for slice_idx in range(num_slices):
                start_qubit = (slice_idx * slice_size) % num_qubits
                # Map numerical indices to custom wire labels
                wires_slice = range(start_qubit, start_qubit + slice_size)

                qml.templates.AngleEmbedding(inputs[start_qubit:start_qubit + slice_size], wires=wires_slice)
                qml.templates.BasicEntanglerLayers(weights, wires=wires_slice)

            # After encoding all slices, measure all used qubits at once
            all_expvals = [qml.expval(qml.PauliZ(wires=w)) for w in range(num_slices * slice_size)]
            return all_expvals

        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        if variational_layer == qml.templates.StronglyEntanglingLayers:
            weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        else:
            weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
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

            # Normalize y_t to be in the range -pi to pi
            y_t = torch.remainder(y_t + torch.pi, 2 * torch.pi) - torch.pi

            y_t = y_t.flatten()

            f_t = torch.sigmoid(self.clayer_out((self.VQC['forget'](y_t)).view(16, 4)))
            i_t = torch.sigmoid(self.clayer_out((self.VQC['input'](y_t)).view(16, 4)))
            g_t = torch.tanh(self.clayer_out((self.VQC['update'](y_t)).view(16, 4)))
            o_t = torch.sigmoid(self.clayer_out((self.VQC['output'](y_t)).view(16, 4)))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

        return h_t

    # Set the seed for reproducibility
    def set_seed(self, seed):
        torch.manual_seed(seed)
