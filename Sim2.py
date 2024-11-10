import itertools
import random
import simpy
import numpy as np

NUM_GENERAL_DOCTORS = 2
NUM_SPECIFIC_DOCTOR = 1
NUM_NURSES = 2
SIM_TIME = 480
ARRIVAL_MEAN = 5
TRIAGE_MEAN = 5
WAITING_ROOM_CAPACITY = 25
INITIAL_PATIENT_MEAN = 20
NUM_RUNS = 100000

class Clinic:
    def __init__(self, env, num_nurses, num_general_doctors, num_specific_doctor, queue_limit):
        self.env = env
        self.nurses = simpy.Resource(env, capacity=num_nurses)
        self.general_doctors = simpy.Resource(env, capacity=num_general_doctors)
        self.specific_doctor = simpy.Resource(env, capacity=num_specific_doctor)
        self.queue = simpy.Container(env, capacity=queue_limit, init=queue_limit)

        self.wait_times_nurse = []      
        self.wait_times_general = []     
        self.wait_times_specific = []    
        self.total_wait_times = []       
        self.queue_lengths = []          
        self.patients_served = {'simple': 0, 'moderate': 0, 'complex': 0}
        self.patients_turned_away = 0    
        self.patients_served_before_420 = 0
        self.patients_served_after_420 = 0
        self.system_times = []

    def treatment(self, patient_type, start_time):
        yield self.queue.get(1)

        nurse_request = self.nurses.request()
        result = yield nurse_request | self.env.timeout(0)
        if nurse_request in result:
            wait_time_nurse = self.env.now - start_time
            self.wait_times_nurse.append(wait_time_nurse)
            triage_time = random.expovariate(1 / TRIAGE_MEAN)
            yield self.env.timeout(triage_time)
            self.nurses.release(nurse_request)
        else:
            nurse_request.cancel()
            doctor_request = self.general_doctors.request()
            result = yield doctor_request | self.env.timeout(0)
            if doctor_request in result:
                wait_time_nurse = self.env.now - start_time
                self.wait_times_nurse.append(wait_time_nurse)
                triage_time = random.expovariate(1 / TRIAGE_MEAN)
                yield self.env.timeout(triage_time)
                self.general_doctors.release(doctor_request)
            else:
                doctor_request.cancel()
                nurse_request = self.nurses.request()
                yield nurse_request
                wait_time_nurse = self.env.now - start_time
                self.wait_times_nurse.append(wait_time_nurse)
                triage_time = random.expovariate(1 / TRIAGE_MEAN)
                yield self.env.timeout(triage_time)
                self.nurses.release(nurse_request)

        total_wait_time = wait_time_nurse

        if patient_type == 'simple':
            self.patients_served['simple'] += 1
            treatment_time = random.expovariate(1 / 15)
            yield self.env.timeout(treatment_time)
            
        elif patient_type == 'moderate':
            self.patients_served['moderate'] += 1
            with self.general_doctors.request() as req:
                yield req
                wait_time_general = self.env.now - (start_time + wait_time_nurse + triage_time)
                self.wait_times_general.append(wait_time_general)
                total_wait_time += wait_time_general
                treatment_time = random.expovariate(1 / 25)
                yield self.env.timeout(treatment_time)
            
        elif patient_type == 'complex':
            self.patients_served['complex'] += 1
            with self.specific_doctor.request() as req:
                yield req
                wait_time_specific = self.env.now - (start_time + wait_time_nurse + triage_time)
                self.wait_times_specific.append(wait_time_specific)
                total_wait_time += wait_time_specific
                treatment_time = random.expovariate(1 / 35)
                yield self.env.timeout(treatment_time)

        total_system_time = self.env.now - start_time
        self.system_times.append(total_system_time)

        self.total_wait_times.append(total_wait_time)
        yield self.queue.put(1)

        if self.env.now < 420:
            self.patients_served_before_420 += 1
        else:
            self.patients_served_after_420 += 1

def patient(env, name, clinic):
    rand_type = random.random()
    if rand_type < 0.6:
        patient_type = 'simple'
    elif rand_type < 0.9:
        patient_type = 'moderate'
    else:
        patient_type = 'complex'

    start_time = env.now

    if clinic.queue.level > 0:
        yield env.process(clinic.treatment(patient_type, start_time))
    else:
        clinic.patients_turned_away += 1

def setup(env, clinic, arrival_mean):
    patient_count = itertools.count()
    initial_patient_count = max(0, int(random.expovariate(1 / INITIAL_PATIENT_MEAN)))

    for _ in range(initial_patient_count):
        env.process(patient(env, f'Initial Patient {next(patient_count)}', clinic))

    while env.now < 420:
        env.process(patient(env, f'Patient {next(patient_count)}', clinic))
        wait_next_arrival = random.expovariate(1 / arrival_mean)
        yield env.timeout(wait_next_arrival)

def run_simulation(seed):
    random.seed(seed)
    env = simpy.Environment()
    clinic = Clinic(env, NUM_NURSES, NUM_GENERAL_DOCTORS, NUM_SPECIFIC_DOCTOR, WAITING_ROOM_CAPACITY)
    env.process(setup(env, clinic, ARRIVAL_MEAN))

    def monitor_queue(env, clinic):
        while True:
            queue_length = WAITING_ROOM_CAPACITY - clinic.queue.level
            clinic.queue_lengths.append((env.now, queue_length))
            yield env.timeout(1)

    env.process(monitor_queue(env, clinic))

    while True:
        env.run(until=env.now + 10)
        if clinic.queue.level == WAITING_ROOM_CAPACITY and env.now > 420:
            break

    service_rate_before_420 = clinic.patients_served_before_420 / 420 if clinic.patients_served_before_420 else 0
    service_rate_after_420 = (clinic.patients_served_before_420 + clinic.patients_served_after_420) / SIM_TIME
    average_system_time = np.mean(clinic.system_times) if clinic.system_times else 0
    average_wait_time_nurse = np.mean(clinic.wait_times_nurse) if clinic.wait_times_nurse else 0
    average_wait_time_general = np.mean(clinic.wait_times_general) if clinic.wait_times_general else 0
    average_wait_time_specific = np.mean(clinic.wait_times_specific) if clinic.wait_times_specific else 0
    average_total_wait_time = np.mean(clinic.total_wait_times) if clinic.total_wait_times else 0

    times_before_420 = [time for time, _ in clinic.queue_lengths if time < 420]
    if times_before_420:
        average_queue_length_before_420 = sum(length for time, length in clinic.queue_lengths if time < 420) / len(times_before_420)
    else:
        average_queue_length_before_420 = 0

    average_total_queue_length = sum(length for _, length in clinic.queue_lengths) / len(clinic.queue_lengths) if clinic.queue_lengths else 0

    return {
        "service_rate_before_420": service_rate_before_420,
        "service_rate_after_420": service_rate_after_420,
        "average_system_time": average_system_time,
        "average_wait_time_nurse": average_wait_time_nurse,
        "average_wait_time_general": average_wait_time_general,
        "average_wait_time_specific": average_wait_time_specific,
        "average_total_wait_time": average_total_wait_time,
        "average_queue_length_before_420": average_queue_length_before_420,
        "average_total_queue_length": average_total_queue_length,
        "patients_served_simple": clinic.patients_served['simple'],
        "patients_served_moderate": clinic.patients_served['moderate'],
        "patients_served_complex": clinic.patients_served['complex'],
        "total_patients_turned_away": clinic.patients_turned_away
    }

all_results = [run_simulation(seed) for seed in range(1, NUM_RUNS + 1)]
averaged_results = {key: np.mean([result[key] for result in all_results]) for key in all_results[0]}

print("\n")
print(f"Service Rate Before 420 Minutes: {averaged_results['service_rate_before_420']:.2f} patients/minute")
print(f"Service Rate After 420 Minutes (Total): {averaged_results['service_rate_after_420']:.2f} patients/minute")
print(f"Average Time in System: {averaged_results['average_system_time']:.2f} minutes")
print(f"Average Wait Time for Nurse: {averaged_results['average_wait_time_nurse']:.2f} minutes")
print(f"Average Wait Time for General Doctors: {averaged_results['average_wait_time_general']:.2f} minutes")
print(f"Average Wait Time for Specific Doctor: {averaged_results['average_wait_time_specific']:.2f} minutes")
print(f"Average Total Wait Time for Patients: {averaged_results['average_total_wait_time']:.2f} minutes")
print(f"Average Queue Length Before 420 Minutes: {averaged_results['average_queue_length_before_420']:.2f}")
print(f"Overall Average Queue Length: {averaged_results['average_total_queue_length']:.2f}")
print(f"Patients Served (Simple): {averaged_results['patients_served_simple']:.2f}")
print(f"Patients Served (Moderate): {averaged_results['patients_served_moderate']:.2f}")
print(f"Patients Served (Complex): {averaged_results['patients_served_complex']:.2f}")
print(f"Total Patients Turned Away: {averaged_results['total_patients_turned_away']:.2f}")
