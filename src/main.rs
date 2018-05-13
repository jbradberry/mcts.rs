
pub trait BoardState {
    fn starting_state() -> Self;

    // fn next_state(&self, action: &'static str) -> Self;

    // fn is_legal(&self, action: &'static str) -> bool;

    // fn legal_actions(&self) -> Vec<String>;

    // fn previous_player(&self) -> usize;

    // fn current_player(&self) -> usize;

    // fn is_ended(&self) -> bool;
}


#[derive(Debug)]
struct ChongState {
    pawn1: u64,
    pawn2: u64,
    stones1: u64,
    stones2: u64,
    next: usize
}


impl BoardState for ChongState {
    fn starting_state() -> Self {
        Self {
            pawn1: 1 << (0 * 8 + 3),
            pawn2: 1 << (7 * 8 + 4),
            stones1: 0,
            stones2: 0,
            next: 1
        }
    }
}


fn main() {
    let start = ChongState::starting_state();
    println!("{:?}", start);
}
