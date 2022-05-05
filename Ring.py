import gm2mt.auxiliary as aux
from gm2mt.Kicker import Kicker

import copy
import numpy as np
import pathlib
import pandas as pd

class Ring:
    def __init__(
        self,
        r_max = 7.157, # radius of the ring's outer edge
        r_min = 7.067, # radius of ring's inner edge
        b_nom = 1.4513, # Nominal magnetic field strength in Tesla
        b_k = Kicker("one turn", 250), # Kicker magnetic field strength in Gauss
        quad_model = "linear", # "linear" "full": what approximation (if any) should be used for the quad fields
        quad_num = 1, # 1, 4: whether quads should be made uniform around the ring (1), or four stations (4)
        n = 0.108, # the "spring constant" associated with the electric quadrupoles in V/m^2
        collimators = "continuous", # discrete or continuous checking for exit out of the ring
        fringe = False # whether fringe effects should be included
    ):
        self.r_max = r_max
        self.r_min = r_min
        self.b_nom = b_nom
        self.quad_model = quad_model
        self.quad_num = quad_num
        self.n = n
        self.collimators = collimators
        self.fringe = fringe

        self.ring_params = copy.deepcopy(vars(self))
        self.b_k = b_k

        self.mode = "simplex"
        loops_list = [(key, len(val)) for key, val in self.ring_params.items() if isinstance(val, (list, np.ndarray))]
        if len(loops_list) == 1:
            self.mode = "multiplex"
        elif len(loops_list) == 0:
            if (isinstance(b_k, list) and len(b_k) != 1) or self.b_k.mp_mode == "multiplex":
                self.mode = "multiplex"
        else:
            raise RuntimeError("Multiplex simulation of multiple changing variables is not supported.")

        self.k_e = aux.n_to_k(n = self.n, quad_num = self.quad_num)
        self.voltage = aux.n_to_voltage(n = self.n, quad_num = self.quad_num)

        if self.mode == "simplex":
            self._verify()
    
    def _verify(self):
        if self.quad_model not in ("linear", "full"):
            raise ValueError(f"Your quadrupole modeling option {self.quad_model} is unrecognized. Please verify it is \"linear\", or \"full\".")
        
        self.quad_num = int(self.quad_num)
        if self.quad_num not in (1, 4):
            raise ValueError(f"Your quad number {self.quad_num} is unrecognized.  Please verify it is 1 or 4.")
        
        if not isinstance(self.n, (float, int, np.float64, np.int64)):
            raise TypeError(f"Your field index {self.n} is not the right type.")

        if self.collimators not in ("continuous", "discrete", "none"):
            raise ValueError(f"Your collimator specification '{self.collimators}' is not recognized.")
        
        if not isinstance(self.fringe, (bool, np.bool_)):
            raise TypeError("Your fringe option is not a boolean!")

    def search(self):
        try:
            mp_var = [(key, val) for key, val in self.ring_params.items() if isinstance(val, (np.ndarray, list))]
            if len(mp_var) == 1:
                var, values = mp_var[0]
                labels = [f"{var}={value}" for value in values]
            elif len(mp_var) == 0:
                if isinstance(self.b_k, list):
                    labels = ["; ".join(f"{key} = {val}" for (key,val) in k.kicker_params.items()) for k in self.b_k]
                elif self.b_k.mp_mode == "multiplex":
                    _, labels = self.b_k.get_labels()
            else:
                raise RuntimeError("Multiplex simulation of multiple changing variables is not supported.")
        except UnboundLocalError:
            raise RuntimeError("It looks like you're trying to do a label search when the ring is in simplex mode...")
        return len(labels), labels

    def _store_parameters(self, full_path):
        with open(full_path + "/params_readable.txt", "a") as file:
            file.write("RING PARAMETERS:\n")
            file.write(f"Ring region: {self.r_min}m - {self.r_max}m\n")
            file.write(f"Nominal field strength: {self.b_nom} T\n")
            file.write(f"Quad modeling: {self.quad_model}\n")
            file.write(f"Quad distribution: {self.quad_num}\n") 
            file.write(f"Quad field index: {self.n}\n")
            file.write(f"Quad k_e: {self.k_e} V/m2\n") #\u00B2
            file.write(f"Quad plate voltage: {self.voltage / 1000} kV\n")
            file.write(f"Collimators: {self.collimators}\n")

            if self.fringe == True:
                file.write(f"Fringe effects: on\n\n")
            else:
                file.write("Fringe effects: off\n\n")

        self._convert_to_lists()
        df = pd.DataFrame(self.ring_params)
        self._unconvert_from_lists()

        try:
            with pd.ExcelWriter(full_path + '/parameters.xlsx', mode = 'a') as writer:
                df.to_excel(writer, sheet_name = 'ring', index = False)
        except FileNotFoundError:
            with pd.ExcelWriter(full_path + '/parameters.xlsx', mode = 'w') as writer:
                df.to_excel(writer, sheet_name = 'ring', index = False)        
        
        self.b_k._store_parameters(full_path)
    
    def _convert_to_lists(self):
        for key, value in self.ring_params.items():
            self.ring_params[key] = [value]

    def _unconvert_from_lists(self):
        for key, value in self.ring_params.items():
            self.ring_params[key] = value[0]

    @classmethod
    def load(cls, dir):
        xl_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + dir + '/parameters.xlsx'

        def convert_to_list(value):
            if isinstance(value, str) and value[0] == "[":
                try:
                    return [float(i) for i in value[1:-1].split(", ")]
                except ValueError:
                    return [val.replace("'", "") for val in value[1:-1].split(", ")]
            else:
                return value

        df = pd.read_excel(xl_path, sheet_name = "ring", header = 0, engine = "openpyxl")
        df_conv = df.applymap(convert_to_list)
        
        return cls(
            r_max = df_conv['r_max'][0],
            r_min = df_conv['r_min'][0],
            b_nom = df_conv['b_nom'][0],
            b_k = Kicker.load(dir),
            quad_model = df_conv['quad_model'][0],
            quad_num = df_conv['quad_num'][0],
            n = df_conv['n'][0],
            collimators = df_conv['collimators'][0],
            fringe = df_conv['fringe'][0].item()
        )


class _MRing(Ring):
    def __init__(self, ring):
        self.ring = ring

    def generate_list(self):
        parameters = self._listify()
        rings = [1] * self.loops
        for i in range(self.loops):
            rings[i] = Ring(
                r_max = parameters['r_max'][i],
                r_min = parameters['r_min'][i],
                b_nom = parameters['b_nom'][i], 
                b_k = parameters['b_k'][i], 
                quad_model = parameters['quad_model'][i], 
                quad_num = parameters['quad_num'][i], 
                n = parameters['n'][i], 
                collimators = parameters['collimators'][i], 
                fringe = parameters['fringe'][i])
        return rings

    def _listify(self):
        parameters_dict = copy.deepcopy(self.ring.ring_params)
        loops_list = [len(param) for param in parameters_dict.values() if isinstance(param, (list, np.ndarray))]
        if isinstance(self.ring.b_k, list):
            loops_list.append(len(self.ring.b_k))
        elif self.ring.b_k.get_loops() is None:
            pass
        else:
            loops_list.append(self.ring.b_k.get_loops())

        if len(loops_list) == 0:
            self.loops = 1
        elif len(loops_list) == 1:
            self.loops = loops_list[0]
        else:
            raise RuntimeError("Multiplex simulations over more than one changing variable is not possible at this time!")
            
        for key, parameter in parameters_dict.items():
            parameters_dict[key] = np.broadcast_to(parameter, self.loops)
        
        if isinstance(self.ring.b_k, list):
            parameters_dict['b_k'] = self.ring.b_k
        else:
            kicker_dict = self.ring.b_k.generate_list(self.loops)
            kicker_objects = []
            for i in range(self.loops):
                kicker_objects.append(Kicker(mode = kicker_dict['mode'][i], B = kicker_dict['B'][i], b_norm = kicker_dict['b_norm'][i], t_norm = kicker_dict['t_norm'][i], kick_max = kicker_dict['kick_max'][i], kicker_num = kicker_dict['kicker_num'][i]))
            parameters_dict['b_k'] = kicker_objects

        return parameters_dict