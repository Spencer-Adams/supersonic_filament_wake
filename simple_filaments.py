import json 
import numpy as np
np.set_printoptions(precision = 4)
import matplotlib.pyplot as plt 

class vortex_filaments:
    """This class contains functions that define vortex filaments and their influence on velocities at certain points"""
    def __init__(self, json_file):
        self.json_file = json_file

    def load_json(self):
        """This function loads the json"""
        with open(self.json_file, 'r') as json_handle:
            input_vals = json.load(json_handle) 
            self.mu_1_pos = input_vals["mu_1_pos"]
            self.mu_2_pos = input_vals["mu_2_pos"]
            self.mu_3_pos = input_vals["mu_3_pos"]
            self.mu_4_pos = input_vals["mu_4_pos"]
            self.mu_positions = np.array([self.mu_1_pos, self.mu_2_pos, self.mu_3_pos, self.mu_4_pos])
            self.mu_1_1 = input_vals["mu_1_1"]
            self.mu_1_2 = input_vals["mu_1_2"]
            self.mu_2_1 = input_vals["mu_2_1"]
            self.mu_2_2 = input_vals["mu_2_2"]
            self.mu_3_1 = input_vals["mu_3_1"]
            self.mu_3_2 = input_vals["mu_3_2"]
            self.mu_4_1 = input_vals["mu_4_1"]
            self.mu_4_2 = input_vals["mu_4_2"]
            self.controls = input_vals["control_points"]
            self.alpha = np.radians(input_vals["alpha[deg]"])
            self.mach_number = input_vals["mach_number"]
            self.dihedral = np.radians(input_vals["dihedral[deg]"])
            self.sweep = np.radians(input_vals["sweep[deg]"])
            self.step_size = input_vals["step_size"]
            self.normal_vec = input_vals["normal_vec"] # this one would change in a more complicated panel configuration. Just doing this to simplify things

    def pull_control_points(self):
        """This function pulls control values from the txt file in the json"""
        control_points = np.empty((5, 3))
        with open(self.controls, 'r') as text_handle:
            points = [list(map(float, line.strip().split())) for line in text_handle]
            # print("Control points:\n", points)  #### important to note, this will fill up one extra spot apparently. 
        for i in range(len(points)-1):
            control_points[i] = points[i]
        self.control_points = np.array(control_points)

    def calc_wake_edge_doublets(self):
        """This function subtracts the bottom doublet from the top one to find the doublet strength at that point"""
        mu_1 = self.mu_1_1-self.mu_1_2
        mu_2 = self.mu_2_1-self.mu_2_2
        mu_3 = self.mu_3_1-self.mu_3_2
        mu_4 = self.mu_4_1-self.mu_4_2
        self.mu_overalls = np.array([mu_1, mu_2, mu_3, mu_4])

    def calc_mu_funcs(self):
        vecs = np.empty((3, 3))  
        mu_positions = self.mu_positions  
        mu_overalls = self.mu_overalls  
        for i in range(len(mu_positions)-1):
            for j in range(len(mu_positions[0])):
                subtraction = mu_positions[i+1][j] - mu_positions[i][j]
                if (subtraction) <= 0.001 and (subtraction) >= -0.001:
                    vecs[i, j] = 0.0
                else:
                    vecs[i, j] = ((mu_overalls[i+1] - mu_overalls[i]) / (mu_positions[i+1][j] - mu_positions[i][j]))
        self.mu_vecs_list = np.array(vecs)
                # print(len(self.mu_positions[0]))
        
    def calc_vortex_dens(self):
        omega_J_1 = np.cross(self.normal_vec, np.gradient(self.mu_vecs_list[0]))
        omega_J_2 = np.cross(self.normal_vec, np.gradient(self.mu_vecs_list[1]))
        omega_J_3 = np.cross(self.normal_vec, np.gradient(self.mu_vecs_list[2]))
        self.omegas = np.array([omega_J_1, omega_J_2, omega_J_3])

    def calc_normal_deltas(self):
        """This function calculates the width of the panel normal to the vortex filament. Multiply the output of this function by the vortex density to get the vortex strength of the filament"""
        normals = []
        for i in range(len(self.mu_positions)-1):
            for j in range(len(self.mu_positions[0])-2):
                delta = np.sqrt((self.mu_positions[i+1][j]-self.mu_positions[i][j])**2+(self.mu_positions[i+1][j+1]-self.mu_positions[i][j+1])**2+(self.mu_positions[i+1][j+2]-self.mu_positions[i][j+2])**2)
                normals.append(delta)
        self.deltas = np.array(normals)

    # def calc_magnitudes(self):
    #     """This function takes in a vector and returns that magnitude of that vector""" # may not be useful for this particular code. 
    #     magnitudes = []
    #     for i in range(len(self.omegas)-1):
    #         for j in range(len(self.omegas)-1):
    #             mag = np.sqrt(np.sum(self.omegas[i, j:j+2]**2 + self.omegas[i+1, j:j+2]**2))
    #             magnitudes.append(mag)
    #     return magnitudes

    def calc_filament_strengths(self): # may need to change this to account for the vortex density not being totally uniform.
        """This function """
        # print("vortex_dens_length", len(self.omegas))
        filament_strengths = []
        for i in range(len(self.omegas)):
            skrength = np.dot(self.omegas[i],self.deltas) # np.dot(self.omegas[i], self.deltas[i])
            filament_strengths.append(skrength)
        self.fil_strength = np.array(filament_strengths)

    def calc_linspace(self):
        num_steps = int((20)/(self.step_size))
        self.x_space = np.linspace(-10,10, num_steps)
        self.y_space = np.linspace(-10,10, num_steps)
        self.z_space = np.linspace(-10,10, num_steps)

    def calc_fil_y_location(self):
        y_loc = np.empty(3)
        for i in range(len(self.mu_positions)-1):
            y_loc[i] = (self.mu_positions[i+1][1] + self.mu_positions[i][1])/2
        self.filament_y_start = np.array(y_loc)

    def calc_cone_singularity_locs(self):
        """This function finds the intersection of the filaments with the mach cones of all the points of interest"""
        intersect_x_y = []
        mach_number = self.mach_number
        fil_y = self.filament_y_start
        B_squared = mach_number**2 - 1
        for i in range(len(self.control_points)):
            for j in range(len(fil_y)):
                fil_x = np.sqrt(B_squared*(((mach_number**2 - 1) * ((fil_y[j]-self.control_points[i][1])**2)))) + self.control_points[i][0]
                if fil_x <= 0.0:
                    # print("x,y", [fil_x, fil_y[j]])
                    # print("at control point: \n", self.control_points[i], "\n")
                    intersect_x_y.append([fil_x, fil_y[j], self.control_points[i][0], self.control_points[i][1], self.control_points[i][2], 0.0, fil_y[j]])
        self.singularities = np.array(intersect_x_y)

    def calc_velocity_influences(self): # in loops, still need to define all the variables in the u, v, and w equations
        mach_number = self.mach_number
        beta_squared = 1 - mach_number**2
        # begin loops to define find the values we need to put in. The limits of integration should be from the panel edges to the singularities at each point. 
        integration_bounds = self.singularities
        velocity_influences = []
        for i in range(len(integration_bounds)):
            k = 1 # assumming supersonic flow. 
            xo = integration_bounds[i][2]
            yo = integration_bounds[i][3]
            zo = integration_bounds[i][4]
            xf = integration_bounds[i][0]
            y = integration_bounds[i][1]
            xi = integration_bounds[i][5]
            u = 0
            # to get more accurate results, change -((self.fil_strength[0]) to -((self.fil_strength[0]*zo) on the v equation. For now, leave off to test sensitivities.  
            v = -((self.fil_strength[0])/(2*np.pi*k))*(((xo-xi)/(np.sqrt(abs((xo-xi)**2 + beta_squared*((yo-y)+zo**2)))))-((xo-xf)/(np.sqrt(abs((xo-xf)**2 + beta_squared*((yo-y)+zo**2))))))*((1)/((yo-y)**2 + zo**2)) 
            w = ((self.fil_strength[0])/(2*np.pi*k))*(((xo-xi)/(np.sqrt(abs((xo-xi)**2 + beta_squared*((yo-y)+zo**2)))))-((xo-xf)/(np.sqrt(abs((xo-xf)**2 + beta_squared*((yo-y)+zo**2))))))*((yo-y)/((yo-y)**2 + zo**2))
            velocity_influences.append([u, v, w, xo, yo, y])
        self.vel_influence = np.array(velocity_influences)

        # print statements to help understand where to get the values from to combine influences later in the function
        print("\n\n[   u       v     w     eval_x   eval_y  filament(named based on y location)]:\n")
        for j in range(len(self.vel_influence)):
            print(self.vel_influence[j], "\n")

        ## Now, sum the influences of all the filaments together.
        combined_inf = []
        epsilon = 0.001
        u_col = self.vel_influence[:, 0]
        v_col = self.vel_influence[:, 1]
        w_col = self.vel_influence[:, 2]
        eval_point_x = self.vel_influence[:, 3]
        eval_point_y = self.vel_influence[:, 4]

        # Start with the first row's influence
        u_combined = u_col[0]
        v_combined = v_col[0]
        w_combined = w_col[0]
        prev_x = eval_point_x[0]
        prev_y = eval_point_y[0]

        for k in range(1, len(self.vel_influence)):
            current_x = eval_point_x[k]
            current_y = eval_point_y[k]

            # refraining from using current_x==prev_x to prevent computer rounding issues. Using an epsilon just in case. 
            if (abs(current_x - prev_x) <= epsilon) and (abs(current_y - prev_y) <= epsilon):
                # Points are equal, combine their influences
                u_combined += u_col[k]
                v_combined += v_col[k]
                w_combined += w_col[k]
            else:
                # Points are different, save the combined influence and update the variables
                combined_inf.append([u_combined, v_combined, w_combined, prev_x, prev_y])
                # You only want to append values to combined_inf once you know there are no more rows you want to combine together.
                u_combined = u_col[k]
                v_combined = v_col[k]
                w_combined = w_col[k]
                prev_x = current_x
                prev_y = current_y

        # Add the last combined influence after the loop finishes
        combined_inf.append([u_combined, v_combined, w_combined, prev_x, prev_y])

        self.combined_influences = np.array(combined_inf)
        # print statements to show the combined influence of all the filaments on all the points (if the filaments lie inside the mach cone of said point)
        print("\n\n[   u       v     w     eval_x   eval_y]:\n")
        for m in range(len(self.combined_influences)):
            print(self.combined_influences[m], "\n")
        
    def plot_vertices(self):
        plt.plot(self.mu_1_pos[0], marker = "o", color = "black", label = "Panel Vertices")
        #plot panel vertices
        plt.plot(self.mu_1_pos[0], self.mu_1_pos[1],self.mu_1_pos[2], marker = "o", color = "black")
        plt.plot(self.mu_2_pos[0], self.mu_2_pos[1],self.mu_1_pos[2], marker = "o", color = "black")
        plt.plot(self.mu_3_pos[0], self.mu_3_pos[1],self.mu_1_pos[2], marker = "o", color = "black")
        plt.plot(self.mu_4_pos[0], self.mu_4_pos[1],self.mu_1_pos[2], marker = "o", color = "black")

    def plot_filaments(self):
        plt.hlines(y = .5, xmin=-10, xmax = 0, color = "blue", label = "Filaments")
        plt.hlines(y = -.5, xmin=-10, xmax = 0, color = "blue")
        plt.hlines(y = -1.5, xmin=-10, xmax = 0, color = "blue")

    def plot_panels(self):
        for i in range(-3, 3):
            plt.hlines(y = i, xmin=0, xmax = .5, color = "black")
        plt.vlines(x = 0, ymin = -3, ymax = 2, color = "black")    

    def plot_control_points(self):  # this is causing some weird graphical stuff to happen. That's why the negative sign has to be on the x portion of the plot there
        j = 0 # THIS LINE AND THE ONE BELOW JUST HELP AVOID CLUTTER IN THE PLOT LEGEND
        plt.plot(self.control_points[j][0], self.control_points[j][1], marker = "o", color = "Red", label = "Control Points")
        for i in range(1, len(self.control_points)):
            # print("control point", i, ":\n", self.control_points[i][0])
            plt.plot(self.control_points[i][0], self.control_points[i][1], marker = "o", color = "Red")            

    def plot_mach_cone(self):
        mach_number = self.mach_number
        y = self.y_space
        B_squared = mach_number**2 - 1
        j = 2 # THIS LINE AND THE ONE BELOW JUST HELP AVOID CLUTTER IN THE PLOT LEGEND
        x = np.sqrt(B_squared*(((mach_number**2 - 1) * ((y-self.control_points[j][1])**2)))) + self.control_points[j][0]
        plt.plot(x,y, color = "Red", label = "Mach Cone")

        ###### comment in or out the lines below in this function to include or disclude all Mach cones in question.
        for i in range(len(self.control_points)):
            x = np.sqrt(B_squared*(((mach_number**2 - 1) * ((y-self.control_points[i][1])**2)))) + self.control_points[i][0]
            plt.plot(x,y, color = "Red")
        ###### comment in or out the lines above in this function to include or disclude all Mach cones in question.

    def run(self):
        self.load_json()
        print("\n mu_positions: \n", self.mu_positions, "\n")

        self.pull_control_points()
        print("control_points:\n", self.control_points, "\n")

        self.calc_wake_edge_doublets()
        print("mu_strengths:\n", self.mu_overalls, "\n")

        self.calc_mu_funcs()
        print("mu_strength_over_position_change:\n", self.mu_vecs_list, "\n")

        self.calc_vortex_dens()
        print("vortex_density:\n", self.omegas, "\n")

        self.calc_normal_deltas()
        print("Span-wise_Panel_lengths (y-length of panels):\n", self.deltas, "\n")

        self.calc_filament_strengths()
        print("filament strengths:\n", self.fil_strength, "\n")

        self.calc_fil_y_location()
        print("Starting_y_filament_locations\n", self.filament_y_start, "\n")

        self.calc_cone_singularity_locs()
        print("[x_sing, y_sing, x_control, y_control, z_control, x_start, y_start]: \n", self.singularities)
        print("Where x_sing=xf, y_sing=y, x_control=xo, y_control=yo, z_control=zo, x_start=xi, and y_start=y(same as y_sing) \n")

        self.calc_velocity_influences()

        self.calc_linspace()
        self.plot_vertices()
        self.plot_filaments()
        self.plot_panels()
        self.plot_control_points()
        self.plot_mach_cone()
        plt.xlim(-5,2)
        plt.ylim(-3.0,2.5)
        plt.legend(loc='upper left')
        plt.show()

if __name__ == "__main__":
    filaments = vortex_filaments("geometry.json")
    filaments.run()
    