/// Block Interleaver for Burst Error Mitigation
/// 
/// Rearranges bits to spread burst errors across FEC codewords
/// Essential for combating multipath fading which causes clustered errors

/// Block interleaver - simple but effective
/// 
/// Input bits written row-by-row, read column-by-column
/// 
/// Example with 16 bits, 4 columns:
/// Input:  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
/// 
/// Write row-wise:
/// 0  1  2  3
/// 4  5  6  7
/// 8  9 10 11
/// 12 13 14 15
/// 
/// Read column-wise:
/// Output: [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
/// 
/// Now if symbols 4-7 are lost (one burst), the errors are at positions 1,5,9,13
/// spread across different FEC blocks!

pub fn interleave(bits: &[u8], num_columns: usize) -> Vec<u8> {
    let n = bits.len();
    
    if num_columns == 0 || n == 0 {
        return bits.to_vec();
    }
    
    let num_rows = (n + num_columns - 1) / num_columns;
    let mut interleaved = vec![0u8; n];
    
    for i in 0..n {
        let row = i / num_columns;
        let col = i % num_columns;
        let output_idx = col * num_rows + row;
        
        if output_idx < n {
            interleaved[output_idx] = bits[i];
        }
    }
    
    interleaved
}

pub fn deinterleave(bits: &[u8], num_columns: usize) -> Vec<u8> {
    let n = bits.len();
    
    if num_columns == 0 || n == 0 {
        return bits.to_vec();
    }
    
    let num_rows = (n + num_columns - 1) / num_columns;
    let mut deinterleaved = vec![0u8; n];
    
    for i in 0..n {
        let col = i / num_rows;
        let row = i % num_rows;
        let input_idx = row * num_columns + col;
        
        if input_idx < n {
            deinterleaved[input_idx] = bits[i];
        }
    }
    
    deinterleaved
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_interleave_deinterleave() {
        let original: Vec<u8> = (0..16).collect();
        let num_columns = 4;
        
        let interleaved = interleave(&original, num_columns);
        let deinterleaved = deinterleave(&interleaved, num_columns);
        
        assert_eq!(original, deinterleaved);
        
        println!("Original:     {:?}", original);
        println!("Interleaved:  {:?}", interleaved);
        println!("Deinterleaved: {:?}", deinterleaved);
    }
    
    #[test]
    fn test_burst_error_spreading() {
        let bits: Vec<u8> = (0..16).collect();
        let interleaved = interleave(&bits, 4);
        
        // Expected: [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
        assert_eq!(interleaved[0], 0);
        assert_eq!(interleaved[1], 4);
        assert_eq!(interleaved[2], 8);
        assert_eq!(interleaved[3], 12);
    }
}
