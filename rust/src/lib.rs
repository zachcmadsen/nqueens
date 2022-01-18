use std::fmt;
use std::iter;

use fastrand::Rng;

const NUM_QUEENS: usize = 8;
const OPTIMAL_FITNESS: usize = 0;
const POPULATION_SIZE: usize = 100;
const TOURNAMENT_SIZE: usize = 10;
const MAX_GENERATIONS: usize = 1000;

/// A single member of a population. The state is encoded in `cols`, a vector
/// where each entry specifies the position of the queen in the corresponding
/// column. For example, `cols[1] = 2` means there's a queen at the
/// intersection of column two and row three (rows are zero-indexed too).
struct Individual {
    cols: Vec<usize>,
}

impl Individual {
    /// Creates and returns a random individual.
    fn random(rng: &Rng) -> Individual {
        let cols = iter::repeat_with(|| rng.usize(0..NUM_QUEENS))
            .take(NUM_QUEENS)
            .collect();
        Individual { cols }
    }

    /// Returns the fitness of the individual. Fitness is measured by the
    /// number of unique conflicts. So the best fitness is zero.
    ///
    /// Storing the fitness as opposed to recomputing it every time would be
    /// more efficient, but it's fast enough as is.
    fn fitness(&self) -> usize {
        let mut fitness = 0;

        for col_i in 0..self.cols.len() {
            for col_j in (col_i + 1)..self.cols.len() {
                let row_i = self.cols[col_i];
                let row_j = self.cols[col_j];
                let on_same_diagonal = row_i as isize - col_i as isize
                    == row_j as isize - col_j as isize
                    || row_i + col_i == row_j + col_j;

                if row_i == row_j {
                    fitness += 1;
                }

                if on_same_diagonal {
                    fitness += 1;
                }
            }
        }

        fitness
    }
}

impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = self.cols.iter().map(|col| col.to_string()).fold(
            String::with_capacity(self.cols.len()),
            |mut acc, b| {
                acc.push_str(&b);
                acc.push(' ');
                acc
            },
        );

        let mut board =
            String::with_capacity(NUM_QUEENS * NUM_QUEENS + NUM_QUEENS - 1);
        for row in 0..NUM_QUEENS {
            for col in 0..NUM_QUEENS {
                if self.cols[col] == row {
                    board.push('Q');
                } else {
                    board.push('.');
                }
            }
            board.push('\n');
        }

        write!(f, "{}\n{}", rows, board)
    }
}

/// Finds and returns the best individual, i.e., the individual with the lowest
/// fitness score.
fn find_best_individual<'a, T>(population: T) -> &'a Individual
where
    T: IntoIterator<Item = &'a Individual>,
{
    // The iterator won't be empty so it's safe to unwrap.
    population.into_iter().min_by_key(|i| i.fitness()).unwrap()
}

/// Returns a random sample of size `amount` from `population`.
///
/// It doesn't handle the case where `amount` is greater than the length of
/// `population`.
///
/// The implementation is based on `choose_multiple` from the `rand` crate:
/// https://docs.rs/rand/latest/src/rand/seq/mod.rs.html.
fn sample<'a>(
    rng: &Rng,
    population: &'a [Individual],
    amount: usize,
) -> Vec<&'a Individual> {
    let mut population_iter = population.iter();
    let mut samples = Vec::with_capacity(amount);
    samples.extend(population_iter.by_ref().take(amount));

    for (i, individual) in population_iter.enumerate() {
        let k = rng.usize(0..(i + 1 + amount));
        if let Some(spot) = samples.get_mut(k) {
            *spot = individual;
        }
    }

    samples
}

/// Selects an individual from `population` using tournament selection.
fn selection<'a>(
    rng: &Rng,
    population: &'a [Individual],
    tournament_size: usize,
) -> &'a Individual {
    let tournament = sample(rng, population, tournament_size);
    find_best_individual(tournament)
}

/// Creates a new individual by performing single point crossover on the two
/// given individuals.
fn crossover(rng: &Rng, a: &Individual, b: &Individual) -> Individual {
    let crossover_point = rng.usize(..a.cols.len());
    let (left_a, _) = a.cols.split_at(crossover_point);
    let (_, right_b) = b.cols.split_at(crossover_point);
    Individual {
        cols: [left_a, right_b].concat(),
    }
}

/// Performs single point mutation on the given individual. On average, one
/// gene will be mutated per individual. A mutated gene is replaced by a random
/// number.
fn mutate(rng: &Rng, individual: &mut Individual) {
    let mutation_chance = 1.0 / NUM_QUEENS as f64;

    for i in 0..individual.cols.len() {
        if rng.f64() < mutation_chance {
            individual.cols[i] = rng.usize(0..NUM_QUEENS);
        }
    }
}

/// Runs a genetic algorithm to find a solution to the n queens problem. It
/// runs until it finds a solution or has reached the max number of
/// generations.
pub fn run() {
    let rng = fastrand::Rng::new();

    let mut generation = 0;
    let mut population: Vec<Individual> =
        iter::repeat_with(|| Individual::random(&rng))
            .take(POPULATION_SIZE)
            .collect();
    let mut best_individual = find_best_individual(&population);
    let mut best_fitness = best_individual.fitness();

    while best_fitness > OPTIMAL_FITNESS && generation < MAX_GENERATIONS {
        let mut new_population = Vec::with_capacity(population.len());

        for _ in 0..population.len() {
            let a = selection(&rng, &population, TOURNAMENT_SIZE);
            let b = selection(&rng, &population, TOURNAMENT_SIZE);
            let mut offspring = crossover(&rng, a, b);

            mutate(&rng, &mut offspring);

            new_population.push(offspring);
        }

        generation += 1;
        population = new_population;
        best_individual = find_best_individual(&population);
        best_fitness = best_individual.fitness();
    }

    println!(
        "generation: {}\nbest fitness: {}\nbest individual:\n{}",
        generation, best_fitness, best_individual
    );
}
