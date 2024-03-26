"""Creates a shift scheduling problem and solves it."""

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model
import pandas as pd
import ast


_OUTPUT_PROTO = flags.DEFINE_string(
    "output_proto", "", "Output file to write the cp_model proto to."
)
_PARAMS = flags.DEFINE_string(
    "params", "max_time_in_seconds:10.0", "Sat solver parameters."
)


def negated_bounded_span(
    works: list[cp_model.BoolVarT], start: int, length: int
) -> list[cp_model.BoolVarT]:
    """Filters an isolated sub-sequence of variables assined to True.

    Extract the span of Boolean variables [start, start + length), negate them,
    and if there is variables to the left/right of this span, surround the span by
    them in non negated form.

    Args:
      works: a list of variables to extract the span from.
      start: the start to the span.
      length: the length of the span.

    Returns:
      a list of variables which conjunction will be false if the sub-list is
      assigned to True, and correctly bounded by variables assigned to False,
      or by the start or end of works.
    """
    sequence = []
    # left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(~works[start + i])
    # right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence


def add_soft_sequence_constraint(
    model: cp_model.CpModel,
    works: list[cp_model.BoolVarT],
    hard_min: int,
    soft_min: int,
    min_cost: int,
    soft_max: int,
    hard_max: int,
    max_cost: int,
    prefix: str,
) -> tuple[list[cp_model.BoolVarT], list[int]]:
    """Sequence constraint on true variables with soft and hard bounds.

    This constraint look at every maximal contiguous sequence of variables
    assigned to true. If forbids sequence of length < hard_min or > hard_max.
    Then it creates penalty terms if the length is < soft_min or > soft_max.

    Args:
      model: the sequence constraint is built on this model.
      works: a list of Boolean variables.
      hard_min: any sequence of true variables must have a length of at least
        hard_min.
      soft_min: any sequence should have a length of at least soft_min, or a
        linear penalty on the delta will be added to the objective.
      min_cost: the coefficient of the linear penalty if the length is less than
        soft_min.
      soft_max: any sequence should have a length of at most soft_max, or a linear
        penalty on the delta will be added to the objective.
      hard_max: any sequence of true variables must have a length of at most
        hard_max.
      max_cost: the coefficient of the linear penalty if the length is more than
        soft_max.
      prefix: a base name for penalty literals.

    Returns:
      a tuple (variables_list, coefficient_list) containing the different
      penalties created by the sequence constraint.
    """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.add_bool_or(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ": under_span(start=%i, length=%i)" % (start, length)
                lit = model.new_bool_var(prefix + name)
                span.append(lit)
                model.add_bool_or(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ": over_span(start=%i, length=%i)" % (start, length)
                lit = model.new_bool_var(prefix + name)
                span.append(lit)
                model.add_bool_or(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.add_bool_or([~works[i] for i in range(start, start + hard_max + 1)])
    return cost_literals, cost_coefficients


def add_soft_sum_constraint(
    model: cp_model.CpModel,
    works: list[cp_model.BoolVarT],
    hard_min: int,
    soft_min: int,
    min_cost: int,
    soft_max: int,
    hard_max: int,
    max_cost: int,
    prefix: str,
) -> tuple[list[cp_model.IntVar], list[int]]:
    """sum constraint with soft and hard bounds.

    This constraint counts the variables assigned to true from works.
    If forbids sum < hard_min or > hard_max.
    Then it creates penalty terms if the sum is < soft_min or > soft_max.

    Args:
      model: the sequence constraint is built on this model.
      works: a list of Boolean variables.
      hard_min: any sequence of true variables must have a sum of at least
        hard_min.
      soft_min: any sequence should have a sum of at least soft_min, or a linear
        penalty on the delta will be added to the objective.
      min_cost: the coefficient of the linear penalty if the sum is less than
        soft_min.
      soft_max: any sequence should have a sum of at most soft_max, or a linear
        penalty on the delta will be added to the objective.
      hard_max: any sequence of true variables must have a sum of at most
        hard_max.
      max_cost: the coefficient of the linear penalty if the sum is more than
        soft_max.
      prefix: a base name for penalty variables.

    Returns:
      a tuple (variables_list, coefficient_list) containing the different
      penalties created by the sequence constraint.
    """
    cost_variables = []
    cost_coefficients = []
    sum_var = model.new_int_var(hard_min, hard_max, "")
    # This adds the hard constraints on the sum.
    model.add(sum_var == sum(works))

    # Penalize sums below the soft_min target.
    if soft_min > hard_min and min_cost > 0:
        delta = model.new_int_var(-len(works), len(works), "")
        model.add(delta == soft_min - sum_var)
        # TODO(user): Compare efficiency with only excess >= soft_min - sum_var.
        excess = model.new_int_var(0, 7, prefix + ": under_sum")
        model.add_max_equality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(min_cost)

    # Penalize sums above the soft_max target.
    if soft_max < hard_max and max_cost > 0:
        delta = model.new_int_var(-7, 7, "")
        model.add(delta == sum_var - soft_max)
        excess = model.new_int_var(0, 7, prefix + ": over_sum")
        model.add_max_equality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(max_cost)

    return cost_variables, cost_coefficients

def load_worker_requests_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return ast.literal_eval(content)

def solve_shift_scheduling(params: str, output_proto: str, workers=None):
    """Solves the shift scheduling problem."""
    # Assuming workers is a dictionary with employee identifiers as keys
    if workers is None:
        workers = {'MG': 0, 'OE': 1, 'YK': 2, 'KA': 3, 'AM': 4, 'AS': 5, 'TC': 6, 'VA': 7, 'NG': 8, 'SL': 9, 'EM': 10,
                   'RN': 11, 'IK': 12, 'TL': 13, 'MT': 14, 'NM': 15, 'OY': 16}
    num_employees = len(workers)
    num_weeks = 5
    shifts = ["O", "M", "E", "N"]

    # Fixed assignment: (employee, shift, day), this fixes the first 2 days of the schedule.
    # Vladi: Should represent the last 2 days of the old Schedule
    fixed_assignments = [
        (0, 2, 0),
        (1, 1, 0),
        (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
        (5, 3, 0),
        (6, 2, 0),
        (7, 3, 0),
        (8, 0, 0),
        (9, 0, 0),
        (10, 0, 0),
        (11, 1, 0),
        (12, 0, 0),
        (13, 0, 0),
        (14, 1, 0),
        (15, 2, 0),
        (16, 0, 0),
        (0, 3, 1),
        (1, 1, 1),
        (2, 2, 1),
        (3, 0, 1),
        (4, 0, 1),
        (5, 2, 1),
        (6, 0, 1),
        (7, 0, 1),
        (8, 0, 1),
        (9, 0, 1),
        (10, 0, 1),
        (11, 3, 1),
        (12, 1, 1),
        (13, 1, 1),
        (14, 2, 1),
        (15, 0, 1),
        (16, 0, 1),
    ]

    fixed_assignments_converted = [
        (list(workers.keys())[e], s, d) for e, s, d in fixed_assignments
    ]

    # Request: (employee, shift, day, weight), a negative weight indicates that the employee desire this assignment.
    requests_file_path = "C:\\Users\\ashur\\Desktop\\WSC\\requests.txt"  # Update with the actual path
    requests = load_worker_requests_from_file(requests_file_path)
    requests_converted = [
        (list(workers.keys())[e], s, d, w) for e, s, d, w in requests
    ]

    # Shift constraints on continuous sequence :
    #     (shift, hard_min, soft_min, min_penalty, soft_max, hard_max, max_penalty)
    shift_constraints = [
        # One or two consecutive days of rest, this is a hard constraint.
       ## (0, 1, 1, 0, 2, 2, 0),
        # between 2 and 3 consecutive days of night shifts, 1 and 4 are possible but penalized.
        (3, 1, 2, 20, 3, 4, 5)
    ]

    # Weekly sum constraints on shifts days:
    #     (shift, hard_min, soft_min, min_penalty,
    #             soft_max, hard_max, max_penalty)
    weekly_sum_constraints = [
        (0, 1, 2, 4, 3, 7, 4)   # Constraints on rests per week.
       # (3, 0, 1, 3, 4, 4, 0), # At least 1 night shift per week (penalized). At most 4 (hard).
    ]

    # Penalized transitions:
    #     (previous_shift, next_shift, penalty (0 means forbidden))
    penalized_transitions = [
        # Night to Morning is forbidden.
        (3, 1, 0),
        # Evening to Morning is penalized.
        (2, 1, 4)  # Assign a penalty value as deemed appropriate.
    ]

    # daily demands for work shifts (morning, afternon, night) for each day of the week starting on Sunday.
    weekly_cover_demands = [
        (2, 2, 2),  # Sunday
        (2, 2, 2),  # Monday
        (2, 2, 2),  # Tuesday
        (2, 2, 2),  # Wednesday
        (2, 2, 2),  # Thursday
        (2, 2, 2),  # Friday
        (2, 2, 2),  # Saturday
    ]

    num_days = num_weeks * 7
    num_shifts = len(shifts)
    model = cp_model.CpModel()

    work = {}

    for e_name in workers.keys():  # Iterate over employee names/identifiers
        for s in range(num_shifts):
            for d in range(num_days):
                # Use the employee's name/identifier in the variable name
                work[e_name, s, d] = model.new_bool_var("work%s_%i_%i" % (e_name, s, d))

    # Linear terms of the objective in a minimization context.
    obj_int_vars: list[cp_model.IntVar] = []
    obj_int_coeffs: list[int] = []
    obj_bool_vars: list[cp_model.BoolVarT] = []
    obj_bool_coeffs: list[int] = []

    # Exactly one shift per day.
    for e_name in workers.keys():  # Iterate over employee names/identifiers
        for d in range(num_days):
            # Use a list comprehension to create a list of variables for the constraint
            model.add_exactly_one(work[e_name, s, d] for s in range(num_shifts))

    # Fixed assignments.
    for e, s, d in fixed_assignments_converted:
        model.add(work[e, s, d] == 1)

    # Employee requests
    for e, s, d, w in requests_converted:
        obj_bool_vars.append(work[e, s, d])
        obj_bool_coeffs.append(w)

    # Shift constraints
    for ct in shift_constraints:
        shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = ct
        for e_name in workers.keys():  # Iterate over employee names/identifiers
            works = [work[e_name, shift, d] for d in range(num_days)]
            variables, coeffs = add_soft_sequence_constraint(
                model,
                works,
                hard_min,
                soft_min,
                min_cost,
                soft_max,
                hard_max,
                max_cost,
                "shift_constraint(employee %s, shift %i)" % (e_name, shift),
            )
            obj_bool_vars.extend(variables)
            obj_bool_coeffs.extend(coeffs)

    # Weekly sum constraints
    for ct in weekly_sum_constraints:
        shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = ct
        for e_name in workers.keys():  # Iterate over employee names/identifiers
            for w in range(num_weeks):
                works = [work[e_name, shift, d + w * 7] for d in range(7)]
                variables, coeffs = add_soft_sum_constraint(
                    model,
                    works,
                    hard_min,
                    soft_min,
                    min_cost,
                    soft_max,
                    hard_max,
                    max_cost,
                    "weekly_sum_constraint(employee %s, shift %i, week %i)"
                    % (e_name, shift, w),
                )
                obj_int_vars.extend(variables)
                obj_int_coeffs.extend(coeffs)


    # Penalized transitions
    for previous_shift, next_shift, cost in penalized_transitions:
        for e_name in workers.keys():  # Use employee names directly
            for d in range(num_days - 1):
                # Use employee names in the work dictionary keys
                transition = [
                    ~work[e_name, previous_shift, d],
                    ~work[e_name, next_shift, d + 1],
                ]
                if cost == 0:
                    # Forbidding the transition by ensuring at least one of the conditions must be false
                    model.add_bool_or(transition)
                else:
                    # Create a new boolean variable for penalized transitions
                    trans_var = model.new_bool_var("transition_%s_day%i_to_day%i" % (e_name, d, d + 1))
                    # Include this variable in the transition condition
                    transition.append(trans_var)
                    # Add the condition to the model
                    model.add_bool_or(transition)
                    # If the transition occurs, this variable will be true, leading to a penalty
                    obj_bool_vars.append(trans_var)
                    obj_bool_coeffs.append(cost)



    # Cover constraints
    max_employees_per_shift = {'M': 4, 'E': 4, 'N': 3}
    for s in range(1, num_shifts):
        shift_type = shifts[s]  # Get the shift type symbol (e.g., 'M', 'E', 'N')
        for w in range(num_weeks):
            for d in range(7):
                # Minimum demand for this shift on this day
                min_demand = weekly_cover_demands[d][s - 1]
                # Generate the list of work variables for all employees for this shift on this day
                works_on_shift = [work[e_name, s, w * 7 + d] for e_name in workers.keys()]
                # Variable for the total number of employees working this shift on this day
                total_worked = model.new_int_var(min_demand, max_employees_per_shift[shift_type],
                                                 f"total_worked_{shift_type}_w{w}_d{d}")
                model.add(total_worked == sum(works_on_shift))

    # Objective
    model.minimize(
        sum(obj_bool_vars[i] * obj_bool_coeffs[i] for i in range(len(obj_bool_vars)))
        + sum(obj_int_vars[i] * obj_int_coeffs[i] for i in range(len(obj_int_vars)))
    )

    if output_proto:
        print("Writing proto to %s" % output_proto)
        with open(output_proto, "w") as text_file:
            text_file.write(str(model))

    # Solve the model.
    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.solve(model, solution_printer)

    # After the solver (if a solution is found):
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

        # Initialize shift_counts
        shift_counts = {}
        for e_name in workers.keys():
            shift_counts[e_name] = {'O': 0, 'M': 0, 'E': 0, 'N': 0, 'Total': 0}
        print()

            # Employee Shift Summary
        print("\nEmployee Shift Summary:")
        for e_name, counts in shift_counts.items():
            for d in range(num_days):
                for s in range(num_shifts):
                    if solver.boolean_value(work[e_name, s, d]):
                        shift_counts[e_name][shifts[s]] += 1
                        if shifts[s] != 'O':  # Only increment 'Total' if not an 'O' shift
                            shift_counts[e_name]['Total'] += 1
            print(f"{e_name}: M = {counts['M']}, E = {counts['E']}, N = {counts['N']}, Total = {counts['Total']}")

            ## Schedule summary to CSV/Print
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Initialize day_shift_employees to organize the schedule
                day_shift_employees = {d: {shift: [] for shift in ['M', 'E', 'N']} for d in range(num_days)}

                # Populate day_shift_employees with the schedule data
                for e_name in workers.keys():
                    for d in range(num_days):
                        for s, shift in enumerate(shifts):  # Assuming shifts = ['O', 'M', 'E', 'N']
                            if shift == 'O':  # Skip if the shift is 'O'
                                continue
                            if solver.boolean_value(work[e_name, s, d]):
                                day_shift_employees[d][shift].append(e_name)

                # Determine the maximum number of employees in any shift on any day
                max_employees_per_shift = max(
                    len(employees) for day in day_shift_employees.values() for employees in day.values())

                # Flatten the shift structure into a list to ensure the correct order in columns
                shift_columns = []
                for i in range(max_employees_per_shift):
                    for shift in ['M', 'E', 'N']:
                        shift_columns.append(f"{shift}{i}")

                # Create a DataFrame with empty strings for all entries
                schedule_df = pd.DataFrame('', index=range(num_days), columns=shift_columns)

                # Populate the DataFrame with employee names
                for d in range(num_days):
                    for shift in ['M', 'E', 'N']:
                        for i, e_name in enumerate(day_shift_employees[d][shift]):
                            column_name = f"{shift}{i}"
                            schedule_df.at[d, column_name] = e_name

                # Insert the Day column at the beginning
                schedule_df.insert(0, 'Day', [f"Day{d}" for d in range(num_days)])

                # Export to CSV
                output_csv_path = "C:\\Users\\ashur\\Desktop\\WSC\\schedule_output.csv"
                schedule_df.to_csv(output_csv_path, index=False)
                print(f"Schedule exported to {output_csv_path}")
                print()

        print()
        print("Penalties:")
        for i, var in enumerate(obj_bool_vars):
            if solver.boolean_value(var):
                penalty = obj_bool_coeffs[i]
                if penalty > 0:
                    print(f"  {var.name} violated, penalty={penalty}")
                else:
                    print(f"  {var.name} fulfilled, gain={-penalty}")

        for i, var in enumerate(obj_int_vars):
            if solver.value(var) > 0:
                print(
                    "  %s violated by %i, linear penalty=%i"
                    % (var.name, solver.value(var), obj_int_coeffs[i])
                )

    print()
    print("Statistics")
    print("  - status          : %s" % solver.status_name(status))
    print("  - conflicts       : %i" % solver.num_conflicts)
    print("  - branches        : %i" % solver.num_branches)
    print("  - wall time       : %f s" % solver.wall_time)


def main(_):
    solve_shift_scheduling(_PARAMS.value, _OUTPUT_PROTO.value)


if __name__ == "__main__":
    app.run(main)