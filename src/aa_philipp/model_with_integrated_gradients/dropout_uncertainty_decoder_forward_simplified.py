

class PSP_Model_simplified:
    def __init__(self, model):
        self.model = model


    def forward(self, prefix):

        prefixes_embedded, final_prefix_element_embedded = prefix

        # ---------- Encoder ----------
        outputs, (h, c), _ = self.model.encoder.first_layer(
            input=prefixes_embedded, hx=None, z=None
        )

        for layer in self.model.encoder.hidden_layers:
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None)

        event = final_prefix_element_embedded


        outputs, (h, c), z_first = self.model.decoder.first_layer(
            input=event, hx=(h, c), z=None
        )
        
        for lstm_cell in self.model.decoder.hidden_layers:
            outputs, (h, c), z_h = lstm_cell(
                input=outputs, hx=(h, c), z=None
            )

        final_output = outputs[-1]

        # ---------- Post-Processing ----------

        cat_output_sizes, num_output_sizes = self.model.decoder.output_sizes

        prediction_means = [{}, {}]
        prediction_vars = [{}, {}]
        
        for key in cat_output_sizes:
            prediction_means[0][f"{key}_mean"] = self.model.decoder.output_layers[f"{key}_mean"](final_output)
            prediction_vars[0][f"{key}_var"]  = self.model.decoder.output_layers[f"{key}_var"](final_output)

        for key in num_output_sizes:
            prediction_means[1][f"{key}_mean"] = self.model.decoder.output_layers[f"{key}_mean"](final_output)
            prediction_vars[1][f"{key}_var"]  = self.model.decoder.output_layers[f"{key}_var"](final_output)

        predictions = [prediction_means, prediction_vars]


        return predictions
