/**
 * Created by Zoe on 13/08/16.
 */


function loadData() {
    var $text = $('#text');

    // clear out old data before new request
    $text.text("");

    // get values
    var startDate = $('#startDate').val();
    var endDate = $('#endDate').val();

    // load data
    $.getJSON($SCRIPT_ROOT+'/cluster', {start_date: startDate, end_date: endDate}, function(data){
        // load json object with distances etc into d3 or chart.js
        $text.text(data.result);
    })

    return false;
};


$('#inputs').submit(loadData);