package org.jetbrains.kotlinx.dl.example.app

import android.content.Context
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import kotlinx.android.synthetic.main.pipelines_selector.view.text1
import kotlinx.android.synthetic.main.pipelines_selector.view.text2

class PipelineSelectorAdapter(context: Context, private val resource: Int, items: List<Pipelines>) :
    ArrayAdapter<Pipelines>(context, resource, items) {
    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        return createView(position, convertView, parent)
    }

    override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
        return createView(position, convertView, parent)
    }

    private fun createView(position: Int, recycledView: View?, parent: ViewGroup): View {
        val view = recycledView ?: LayoutInflater.from(context).inflate(
            resource,
            parent,
            false
        )

        val pipeline = getItem(position)
        view.text1.text = pipeline?.descriptionId?.let { context.getString(it) } ?: ""
        view.text2.text = pipeline?.task?.descriptionId?.let { context.getString(it) } ?: ""

        return view
    }
}