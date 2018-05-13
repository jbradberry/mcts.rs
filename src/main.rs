
pub trait BoardState {
    fn starting_state() -> Self;

    fn previous_player(&self) -> usize;

    fn current_player(&self) -> usize;

    // fn next_state(&self, action: &ChongAction) -> Self;

    // fn is_legal(&self, action: &ChongAction) -> bool;

    // fn legal_actions(&self) -> Vec<ChongAction>;

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


#[derive(Debug)]
enum ChongPiece {
    Pawn,
    Stone
}


#[derive(Debug)]
struct ChongAction {
    piece: ChongPiece,
    x: u8,
    y: u8
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

    fn previous_player(&self) -> usize {
        3 - self.next
    }

    fn current_player(&self) -> usize {
        self.next
    }
}


fn main() {
    let start = ChongState::starting_state();
    println!("{:?}", start);
    println!("{}", start.current_player());
    println!("{}", start.previous_player());
}
