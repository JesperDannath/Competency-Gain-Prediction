import sys
import os
sys.path.append(os.path.realpath("./models"))
if True:  # noqa: E402
    import simulation_experiment

result_dict = simulation_experiment.mirt_simulation_experiment(
                sample_size=30, item_dimension=10, latent_dimension=2, 
                q_type="full", 
                early_person_method="BFGS", late_person_method="BFGS",
                methods=["real_early", "pure_competency", "initial",
                            "late_em", "difference", "real_parameters"],
                gain_mean=1.5)

print("Late EM, Initial Competency Covariance:")
print(result_dict["late_em"]["estimated_early_parameters"]["person_parameters"]["covariance"])
print("Late EM, Full Covariance:")
print(result_dict["late_em"]["estimated_late_parameters"]["person_parameters"]["covariance"])